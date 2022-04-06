import torch.nn as nn
import torch
import torch.nn.functional as F
from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.hrnet_ocr import HighResolutionNet
from isegm.model.modifiers import LRMult
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
import torchvision.ops.roi_align as roi_align
from isegm.model.ops import DistMaps

class HRNetModel(ISModel):
    @serialize
    def __init__(self, width=48, ocr_width=256, small=True, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, pipeline_version = 's1',
                 **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.feature_extractor = HighResolutionNet(width=width, ocr_width=ocr_width, small=small,
                                                   num_classes=1, norm_layer=norm_layer)
        self.feature_extractor.apply(LRMult(backbone_lr_mult))
        if ocr_width > 0:
            self.feature_extractor.ocr_distri_head.apply(LRMult(1.0))
            self.feature_extractor.ocr_gather_head.apply(LRMult(1.0))
            self.feature_extractor.conv3x3_ocr.apply(LRMult(1.0))
        
        self.feature_extractor.apply(LRMult(2))
        self.width=width
        self.pipeline_version = pipeline_version
        if self.pipeline_version == 's1':
            base_radius = 2
        else:
            base_radius = 5
        
        self.refiner = RefineLayer(feature_dims=ocr_width * 2)
        
        self.dist_maps_base = DistMaps(norm_radius=base_radius, spatial_scale=1.0,
                                      cpu_mode=False, use_disks=True)
        
        self.dist_maps_refine = DistMaps(norm_radius=5, spatial_scale=1.0,
                                      cpu_mode=False, use_disks=True)
                                    
    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps_base(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        return coord_features
    
    def backbone_forward(self, image, coord_features=None):
        mask, mask_aux, feature = self.feature_extractor(image, coord_features)
        return {'instances': mask, 'instances_aux':mask_aux, 'feature': feature}

    def refine(self, cropped_image, cropped_points, full_feature, full_logits, bboxes):
        '''
        bboxes : [b,5]
        '''
        h1 = cropped_image.shape[-1]
        h2 = full_feature.shape[-1]
        r = h1/h2

        cropped_feature = roi_align(full_feature,bboxes,full_feature.size()[2:], spatial_scale=1/r, aligned = True)
        cropped_logits = roi_align(full_logits,bboxes,cropped_image.size()[2:], spatial_scale=1, aligned = True)
        
        click_map = self.dist_maps_refine( cropped_image,cropped_points)
        refined_mask, trimap = self.refiner(cropped_image,click_map,cropped_feature,cropped_logits)
        return {'instances_refined': refined_mask, 'instances_coarse':cropped_logits, 'trimap':trimap}

    def forward(self, image, points):
        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)
        click_map = coord_features[:,1:,:,:]

        if self.pipeline_version == 's1':
            small_image = F.interpolate(image, scale_factor=0.5,mode='bilinear',align_corners=True)
            small_coord_features = F.interpolate(coord_features, scale_factor=0.5,mode='bilinear',align_corners=True)
        else:
            small_image = image
            small_coord_features = coord_features

        small_coord_features = self.maps_transform(small_coord_features)
        outputs = self.backbone_forward( small_image, small_coord_features)

        outputs['click_map'] = click_map
        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)
        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                             mode='bilinear', align_corners=True)
        return outputs





class RefineLayer(nn.Module):
    """
    Refine Layer for Full Resolution
    """
    def __init__(self, input_dims = 6, mid_dims = 32, feature_dims = 96, num_classes = 1,  **kwargs):
        super(RefineLayer, self).__init__()
        self.num_classes = num_classes
        self.image_conv1 = ConvModule(
            in_channels=input_dims,
            out_channels= mid_dims,
            kernel_size=3,
            stride=2,
            padding=1,
            #norm_cfg=dict(type='BN', requires_grad=True),
            )
        self.image_conv2 = XConvBnRelu( mid_dims, mid_dims)
        self.image_conv3 = XConvBnRelu( mid_dims, mid_dims)
        

        self.refine_fusion = ConvModule(
            in_channels= feature_dims,
            out_channels= mid_dims,
            kernel_size=1,
            stride=1,
            padding=0,
            #norm_cfg=dict(type='BN', requires_grad=True),
            )
        
        self.refine_fusion2 = XConvBnRelu( mid_dims, mid_dims)
        self.refine_fusion3 = XConvBnRelu( mid_dims, mid_dims)
        self.refine_pred = nn.Conv2d( mid_dims, num_classes,3,1,1)
        self.refine_trimap = nn.Conv2d( mid_dims, num_classes,3,1,1)
    

    def forward(self, input_image, click_map, final_feature, cropped_logits):
        #cropped_logits = cropped_logits.clone().detach()
        #final_feature = final_feature.clone().detach()

        mask = cropped_logits #resize(cropped_logits, size=input_image.size()[2:],mode='bilinear',align_corners=True)
        bin_mask = torch.sigmoid(mask) #> 0.49
        input_image = torch.cat( [input_image,click_map,bin_mask], 1)

        final_feature = self.refine_fusion(final_feature)
        image_feature = self.image_conv1(input_image)
        image_feature = self.image_conv2(image_feature)
        image_feature = self.image_conv3(image_feature)
        pred_feature = resize(final_feature, size=image_feature.size()[2:],mode='bilinear',align_corners=True)
        #fusion_gate = self.feature_gate(final_feature)
        #fusion_gate = resize(fusion_gate, size=image_feature.size()[2:],mode='bilinear',align_corners=True)
        pred_feature = pred_feature + image_feature #* fusion_gate

        pred_feature = self.refine_fusion2(pred_feature)
        pred_feature = self.refine_fusion3(pred_feature)
        pred_full = self.refine_pred(pred_feature)
        trimap = self.refine_trimap(pred_feature)
        trimap = F.interpolate(trimap, size=input_image.size()[2:],mode='bilinear',align_corners=True)
        pred_full = F.interpolate(pred_full, size=input_image.size()[2:],mode='bilinear',align_corners=True)
        trimap_sig = torch.sigmoid(trimap)
        pred = pred_full * trimap_sig + mask * (1-trimap_sig)
        return pred, trimap

class ConvModule(nn.Module):
    def __init__(self, in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
       

    def forward(self, x):
        return self.activation( self.norm( self.conv(x)  ) )




class XConvBnRelu(nn.Module):
    """
    Xception conv bn relu
    """
    def __init__(self, input_dims = 3, out_dims = 16,   **kwargs):
        super(XConvBnRelu, self).__init__()
        self.conv3x3 = nn.Conv2d(input_dims,input_dims,3,1,1,groups=input_dims)
        self.conv1x1 = nn.Conv2d(input_dims,out_dims,1,1,0)
        self.norm = nn.BatchNorm2d(out_dims)
        self.activation = nn.ReLU()
    def forward(self,x):
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        x = self.norm(x)
        x = self.activation(x)
        return x





def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           ):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)