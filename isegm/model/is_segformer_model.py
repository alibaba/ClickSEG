import torch.nn as nn
import torch
import torch.nn.functional as F
from isegm.utils.serialization import serialize
from .is_model import ISModel
from isegm.model.ops import DistMaps, ScaleLayer, BatchImageNormalize
from .modeling.hrnet_ocr import HighResolutionNet
from isegm.model.modifiers import LRMult
from .modeling.segformer.segformer_model import SegFormer
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
import torchvision.ops.roi_align as roi_align
from isegm.model.ops import DistMaps


class SegFormerModel(ISModel):
    @serialize
    def __init__(self, feature_stride = 4, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, pipeline_version = 's1', model_version = 'b0',
                  **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.pipeline_version = pipeline_version
        self.model_version = model_version
        self.feature_extractor = SegFormer(self.model_version)
        self.feature_extractor.backbone.apply(LRMult(backbone_lr_mult))

        if self.pipeline_version == 's1':
            base_radius = 3
        else:
            base_radius = 5
        
        self.dist_maps_base = DistMaps(norm_radius=base_radius, spatial_scale=1.0,
                                      cpu_mode=False, use_disks=True)
        
        self.dist_maps_refine = DistMaps(norm_radius=5, spatial_scale=1.0,
                                      cpu_mode=False, use_disks=True)

        if self.model_version == 'b0':
            feature_indim = 256
        else:
            feature_indim = 512
        self.refiner = RefineLayer(feature_indim = feature_indim)
        
        mt_layers = [
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                ScaleLayer(init_value=0.05, lr_mult=1)
            ]
        self.maps_transform = nn.Sequential(*mt_layers)

    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps_base(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        return coord_features


    def backbone_forward(self, image, side_feature):
        mask, feature = self.feature_extractor(image, side_feature)
        return {'instances': mask, 'instances_aux':mask, 'feature': feature}
        

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
        return {'instances_refined': refined_mask, 'trimap':trimap, 'instances_coarse':cropped_logits}

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

        #small_coord_features = self.maps_transform(small_coord_features)
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
    def __init__(self, input_dims = 6, feature_indim = 256, feature_dims = 32, num_classes = 1,  **kwargs):
        super(RefineLayer, self).__init__()
        self.num_classes = num_classes
        self.image_conv1 = ConvModule(
            in_channels=input_dims,
            out_channels=feature_dims,
            kernel_size=3,
            stride=2,
            padding=1,
            )
        self.image_conv2 = XConvBnRelu(feature_dims,feature_dims)
        self.image_conv3 = XConvBnRelu(feature_dims,feature_dims)
        

        self.refine_fusion = ConvModule(
            in_channels= feature_indim,
            out_channels=feature_dims,
            kernel_size=1,
            stride=1,
            padding=0,
            #norm_cfg=dict(type='BN', requires_grad=True),
            )
        
        self.refine_fusion2 = XConvBnRelu(feature_dims,feature_dims)
        self.refine_fusion3 = XConvBnRelu(feature_dims,feature_dims)
        self.feature_gate = nn.Sequential(
            nn.Conv2d(feature_dims, 1, 1),
            nn.Sigmoid()
        )
        self.refine_pred = nn.Conv2d(feature_dims, num_classes,3,1,1)
        self.refine_trimap = nn.Conv2d(feature_dims, num_classes,3,1,1)
    

    def forward(self, input_image, click_map, final_feature, cropped_logits):

        mask = cropped_logits #resize(cropped_logits, size=input_image.size()[2:],mode='bilinear',align_corners=True)
        bin_mask = torch.sigmoid(mask) #> 0.49
        input_image = torch.cat( [input_image,click_map,bin_mask], 1)

        final_feature = self.refine_fusion(final_feature)
        image_feature = self.image_conv1(input_image)
        image_feature = self.image_conv2(image_feature)
        image_feature = self.image_conv3(image_feature)
        pred_feature = resize(final_feature, size=image_feature.size()[2:],mode='bilinear',align_corners=True)
        pred_feature = pred_feature + image_feature #* fusion_gate

        pred_feature = self.refine_fusion2(pred_feature)
        pred_feature = self.refine_fusion3(pred_feature)
        pred_full = self.refine_pred(pred_feature)
        trimap = self.refine_trimap(pred_feature)
        trimap = F.interpolate(trimap, size=input_image.size()[2:],mode='bilinear',align_corners=True)
        pred_full = F.interpolate(pred_full, size=input_image.size()[2:],mode='bilinear',align_corners=True)
        trimap_sig = torch.sigmoid(trimap)
        #trimap_sig = (trimap_sig > 0.8) * 0.99
        #trimap_sig[trimap_sig>0.7] = 1.0
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