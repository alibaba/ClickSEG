import argparse

from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string

#from mmseg.models import build_segmentor
import torch
from isegm.model.modeling.segformer.segformer_model import SegFormer
from isegm.model.is_segformer_model import RefineLayer
from isegm.model.is_segformer_model_inception import RefineLayer as RefineLayer_X
from isegm.model.is_hrnet_model import HRNetModel
import torch.nn as nn
from isegm.model.modeling.hrnet_ocr import HighResolutionNet
from isegm.model.modeling.deeplab_v3 import DeepLabV3Plus

segformer_model = SegFormer(backbone='b3')
deeplab = DeepLabV3Plus(backbone='resnet101', ch=128, project_dropout=0.1,norm_layer=nn.BatchNorm2d, backbone_norm_layer=nn.BatchNorm2d)
refine_layer = RefineLayer_X()
#hr18s_ocr = HRNetModel(width=18, ocr_width=48, small=True, with_aux_output=True, use_leaky_relu=True,
#                       use_rgb_conv=False, use_disks=True, norm_radius=5, binary_prev_mask= False,
#                       with_prev_mask=True)
hr18s_ocr = HighResolutionNet(width=18, ocr_width=48, small=True, num_classes=1, norm_layer=nn.BatchNorm2d)
hr32_ocr = HighResolutionNet(width=32, ocr_width=128, num_classes=1, norm_layer=nn.BatchNorm2d)

def sra_flops(h, w, r, dim, num_heads):
    dim_h = dim / num_heads
    n1 = h * w
    n2 = h / r * w / r

    f1 = n1 * dim_h * n2 * num_heads
    f2 = n1 * n2 * dim_h * num_heads

    return f1 + f2


def get_tr_flops(net, input_shape, input_cons = None):
    flops, params = get_model_complexity_info(net, input_shape, as_strings=False, input_constructor=input_cons)
    _, H, W = input_shape
    net = net.backbone
    try:
        stage1 = sra_flops(H // 4, W // 4,
                           net.block1[0].attn.sr_ratio,
                           net.block1[0].attn.dim,
                           net.block1[0].attn.num_heads) * len(net.block1)
        stage2 = sra_flops(H // 8, W // 8,
                           net.block2[0].attn.sr_ratio,
                           net.block2[0].attn.dim,
                           net.block2[0].attn.num_heads) * len(net.block2)
        stage3 = sra_flops(H // 16, W // 16,
                           net.block3[0].attn.sr_ratio,
                           net.block3[0].attn.dim,
                           net.block3[0].attn.num_heads) * len(net.block3)
        stage4 = sra_flops(H // 32, W // 32,
                           net.block4[0].attn.sr_ratio,
                           net.block4[0].attn.dim,
                           net.block4[0].attn.num_heads) * len(net.block4)
    except:
        stage1 = sra_flops(H // 4, W // 4,
                           net.block1[0].attn.squeeze_ratio,
                           64,
                           net.block1[0].attn.num_heads) * len(net.block1)
        stage2 = sra_flops(H // 8, W // 8,
                           net.block2[0].attn.squeeze_ratio,
                           128,
                           net.block2[0].attn.num_heads) * len(net.block2)
        stage3 = sra_flops(H // 16, W // 16,
                           net.block3[0].attn.squeeze_ratio,
                           320,
                           net.block3[0].attn.num_heads) * len(net.block3)
        stage4 = sra_flops(H // 32, W // 32,
                           net.block4[0].attn.squeeze_ratio,
                           512,
                           net.block4[0].attn.num_heads) * len(net.block4)

    print(stage1 + stage2 + stage3 + stage4)
    flops += stage1 + stage2 + stage3 + stage4
    return flops_to_string(flops), params_to_string(params)




def prepare_input(resolution):
    image = torch.FloatTensor(1, *resolution)
    B,C,H,W = image.shape
    click_map = torch.FloatTensor(1, 2, H, W)
    final_feature = torch.FloatTensor(1, 256, H//4, W//4)
    cropped_logits = torch.FloatTensor(1, 1, H, W)
    return dict(x = image, x1 = click_map, x2 = final_feature, x3=cropped_logits)


'''
flops_seg, params_seg = get_tr_flops(segformer_model, (3,256,256))
flops_refine, params_refine = get_model_complexity_info(refine_layer, (3,256,256))
print('segmentor: ',flops_seg, params_seg)
print('refiner: ',flops_refine, params_refine)
'''


'''
flops_seg, params_seg = get_model_complexity_info(hr18s_ocr, (3,600,600))
flops_refine, params_refine = get_model_complexity_info(refine_layer, (3,256,256))
print('segmentor: ',flops_seg, params_seg)
print('refiner: ',flops_refine, params_refine)
'''

'''
flops_seg, params_seg = get_model_complexity_info(deeplab, (3,512,512))
print('segmentor: ',flops_seg, params_seg)
'''

flops_seg, params_seg = get_model_complexity_info(hr32_ocr, (3,256,256))
#flops_refine, params_refine = get_model_complexity_info(refine_layer, (3,256,256))
print('segmentor: ',flops_seg, params_seg)