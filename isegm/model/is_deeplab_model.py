import torch.nn as nn
import torch.nn.functional as F
import torch
from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.deeplab_v3 import DeepLabV3Plus
from .modeling.basic_blocks import SepConvHead
from isegm.model.modifiers import LRMult
from isegm.model.modeling.cdnet.FDM import FDM
from isegm.model.modeling.cdnet.PDM import PDM
from isegm.model.ops import DistMaps
import math

class DeeplabModel(ISModel):
    @serialize
    def __init__(self, backbone='resnet50', deeplab_ch=256, aspp_dropout=0.5,
                 backbone_norm_layer=None, backbone_lr_mult=0.1, norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.feature_extractor = DeepLabV3Plus(backbone=backbone, ch=deeplab_ch, project_dropout=aspp_dropout,
                                               norm_layer=norm_layer, backbone_norm_layer=backbone_norm_layer)
        self.feature_extractor.backbone.apply(LRMult(backbone_lr_mult))
        self.head = SepConvHead(1, in_channels=deeplab_ch, mid_channels=deeplab_ch // 2,
                                num_layers=2, norm_layer=norm_layer)
        self.latent_head = SepConvHead(4, in_channels=deeplab_ch, mid_channels=deeplab_ch // 2,
                                num_layers=2, norm_layer=norm_layer)
        self.dist_maps = DistMaps(norm_radius=5, spatial_scale=1.0,cpu_mode=False, use_disks=True)
        self.fdm_dist_maps = DistMaps(norm_radius=24, spatial_scale=1.0,cpu_mode=False, use_disks=True)
        self.PDM = PDM()

    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps(image, points)
        fdm_clicks = self.fdm_dist_maps(image, points)
        h,w = fdm_clicks.shape[-2],fdm_clicks.shape[-1]
        hs,ws = math.ceil(h/8),math.ceil(w/8)
        fdm_clicks = F.interpolate(fdm_clicks,(hs,ws),mode='bilinear',align_corners=True)

        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        
        return coord_features, fdm_clicks


    def backbone_forward(self, image, coord_features, fdm_clicks):
        backbone_features, pos_map = self.feature_extractor(image, coord_features,fdm_clicks)
        #detached_feature = backbone_features.clone().detach()
        #pred = self.head(detached_feature)
        pred = self.head(backbone_features)
        latent_preds = self.latent_head(backbone_features)
        return {'instances': pred, 'fdm_instances': pos_map, 'latent_instances':latent_preds}

    
    def forward(self, image, points):
        image, prev_mask = self.prepare_input(image)
        coord_features, fdm_clicks = self.get_coord_features(image, prev_mask, points)


        if coord_features.shape[1] == 3:
            click_map = coord_features[:,1:,:,:]
        else:
            click_map = coord_features
        small_image = image
        small_coord_features = coord_features

        small_coord_features = self.maps_transform(small_coord_features)
        outputs = self.backbone_forward( small_image, small_coord_features, fdm_clicks)

        outputs['click_map'] = click_map
        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)
        outputs['fdm_instances'] = nn.functional.interpolate(outputs['fdm_instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)
        outputs['latent_instances'] = nn.functional.interpolate(outputs['latent_instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)                               
        return outputs
    
    def PixelDiffusion(self, image, mask, clickmap):
        instance_out = self.PDM(image, mask, clickmap)
        return instance_out 
        
