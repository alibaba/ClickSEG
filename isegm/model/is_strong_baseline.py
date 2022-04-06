import torch.nn as nn
import torch
import torch.nn.functional as F
from isegm.utils.serialization import serialize
from .is_model import ISModel
from isegm.model.ops import DistMaps, ScaleLayer, BatchImageNormalize
from isegm.model.modeling.ppclnet.segmentator import PPLCSeg
from isegm.model.modeling.mobilenet.segmentator import MobileSeg
from isegm.model.modeling.efficientnet.segmentator import EfficientSeg
from isegm.model.modifiers import LRMult
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
import torchvision.ops.roi_align as roi_align
from isegm.model.ops import DistMaps


class BaselineModel(ISModel):
    @serialize
    def __init__(self, backbone = 'PPLCNet', backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)
        if backbone == 'PPLCNet':
            self.feature_extractor = PPLCSeg()
            side_feature_ch = 32
        elif backbone == 'MobleNetV2':
            self.feature_extractor = MobileSeg()
            side_feature_ch = 32
        elif backbone == 'EfficientNet':
            self.feature_extractor = EfficientSeg()
            side_feature_ch = 32

        self.feature_extractor.apply(LRMult(backbone_lr_mult))
        self.dist_maps = DistMaps(norm_radius=5, spatial_scale=1.0,
                                      cpu_mode=False, use_disks=True)

        mt_layers = [
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(in_channels=16, out_channels=side_feature_ch, kernel_size=3, stride=1, padding=1),
                ScaleLayer(init_value=0.05, lr_mult=1)
            ]
        self.maps_transform = nn.Sequential(*mt_layers)


    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        return coord_features
    
    def backbone_forward(self, image, coord_features=None):
        mask, mask_aux, feature = self.feature_extractor(image, coord_features)
        return {'instances': mask, 'instances_aux':mask_aux, 'feature': feature}


    def forward(self, image, points):
        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)
        click_map = coord_features[:,1:,:,:]

        coord_features = self.maps_transform(coord_features)
        outputs = self.backbone_forward(image, coord_features)

        outputs['click_map'] = click_map
        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)
        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                             mode='bilinear', align_corners=True)
        return outputs





