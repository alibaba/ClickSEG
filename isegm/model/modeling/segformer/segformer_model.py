from isegm.model.modeling.segformer.mix_transformer import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from isegm.model.modeling.segformer.segformer_head import SegFormerHead
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import ExitStack



class SegFormer(nn.Module):
    def __init__(self, backbone='b0', norm_layer=nn.BatchNorm2d,
                 inference_mode=False,
                 **kwargs):
        super(SegFormer, self).__init__()
        self.backbone_name = backbone


        self._kwargs = kwargs
        if backbone == 'b0':
            in_channels=[32, 64, 160, 256]
            in_index=[0, 1, 2, 3]
            feature_strides=[4, 8, 16, 32]
            #channels=128
            dropout_ratio=0.1
            num_classes = 1
            decoder_params=dict(embed_dim=256)
            self.backbone = mit_b0()
            self.decode_head = SegFormerHead(feature_strides=feature_strides, in_channels=in_channels, 
                                             num_classes=num_classes,in_index=in_index, dropout_ratio = dropout_ratio,
                                              decoder_params = decoder_params)
        elif backbone == 'b3':
            in_channels=[64, 128, 320, 512]
            in_index=[0, 1, 2, 3]
            feature_strides=[4, 8, 16, 32]
            dropout_ratio=0.1
            num_classes = 1
            decoder_params=dict(embed_dim=512)
            self.backbone = mit_b3()
            self.decode_head = SegFormerHead(feature_strides=feature_strides, in_channels=in_channels, 
                                             num_classes=num_classes,in_index=in_index, dropout_ratio = dropout_ratio,
                                              decoder_params = decoder_params)
        
        elif backbone == 'b5':
            in_channels=[64, 128, 320, 512]
            in_index=[0, 1, 2, 3]
            feature_strides=[4, 8, 16, 32]
            #channels=128
            dropout_ratio=0.1
            num_classes = 1
            decoder_params=dict(embed_dim=512)
            self.backbone = mit_b5()
            self.decode_head = SegFormerHead(feature_strides=feature_strides, in_channels=in_channels, 
                                             num_classes=num_classes,in_index=in_index, dropout_ratio = dropout_ratio,
                                              decoder_params = decoder_params)
        
        self.inference_mode = inference_mode
        if inference_mode:
            self.set_prediction_mode()

    def load_pretrained_weights(self, path_to_weights= ' '):    
        backbone_state_dict = self.backbone.state_dict()
        pretrained_state_dict = torch.load(path_to_weights, map_location='cpu')
        ckpt_keys = set(pretrained_state_dict.keys())
        own_keys = set(backbone_state_dict.keys())
        missing_keys = own_keys - ckpt_keys
        unexpected_keys = ckpt_keys - own_keys
        print('Missing Keys: ', missing_keys)
        print('Unexpected Keys: ', unexpected_keys)
        backbone_state_dict.update(pretrained_state_dict)
        self.backbone.load_state_dict(backbone_state_dict, strict= False)

        if self.inference_mode:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def set_prediction_mode(self):
        self.inference_mode = True
        self.eval()

    def forward(self, x, additional_features=None):
        with ExitStack() as stack:
            if self.inference_mode:
                stack.enter_context(torch.no_grad())

            features = self.backbone(x, additional_features)
            pred, feature = self.decode_head(features)
        return pred,feature

