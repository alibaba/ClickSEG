'''
Code Adopted from https://github.com/Tramac/mobilenetv3-segmentation/blob/master/core/model/segmentation.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from isegm.model.modeling.efficientnet.backbone import efficientnet_b0



class LRASPP(nn.Module):
    """Lite R-ASPP"""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(LRASPP, self).__init__()
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )
        self.b1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((2,2)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat2 = F.interpolate(feat2, size, mode='bilinear', align_corners=True)
        x = feat1 * feat2  
        return x



class EfficientSeg(nn.Module):
    def __init__(self, nclass=1, **kwargs):
        super(EfficientSeg, self).__init__()
        self.backbone = efficientnet_b0()
        self.lraspp = LRASPP(320,128)
        self.fusion_conv1 = nn.Conv2d(128,16,1,1,0)
        self.fusion_conv2 = nn.Conv2d(24,16,1,1,0)
        self.head = nn.Conv2d(16,nclass,1,1,0)
        self.aux_head = nn.Conv2d(16,nclass,1,1,0)

    def forward(self, x, side_feature):
        x4, _, _, x8 = self.backbone(x, side_feature)
        x8 = self.lraspp(x8)
        x8 = F.interpolate(x8, x4.size()[2:], mode='bilinear', align_corners=True)
        x8 = self.fusion_conv1(x8)
        pred_aux = self.aux_head(x8)

        x4 = self.fusion_conv2(x4)
        x = x4 + x8
        pred = self.head(x)
        return pred, pred_aux, x

    def load_pretrained_weights(self, path_to_weights= ' '):    
        backbone_state_dict = self.backbone.state_dict()
        pretrained_state_dict = torch.load(path_to_weights, map_location='cpu')
        ckpt_keys = set(pretrained_state_dict.keys())
        own_keys = set(backbone_state_dict.keys())
        missing_keys = own_keys - ckpt_keys
        unexpected_keys = ckpt_keys - own_keys
        print('Loading Efficientnet')
        print('Missing Keys: ', missing_keys)
        print('Unexpected Keys: ', unexpected_keys)
        backbone_state_dict.update(pretrained_state_dict)
        self.backbone.load_state_dict(backbone_state_dict, strict= False)