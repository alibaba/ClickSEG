import torch
import torch.nn as nn
from termcolor import cprint
import torch.nn.functional as F
from isegm.model.modeling.basic_blocks import SepConvHead, ConvHead


class SpatialNL(nn.Module):
    """Spatial NL block for image classification.
       [https://github.com/facebookresearch/video-nonlocal-net].
    """
    def __init__(self, inplanes, planes, use_scale=False):
        self.use_scale = use_scale

        super(SpatialNL, self).__init__()
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(inplanes)

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h, w = t.size()

        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)
        g = g.view(b, c, -1).permute(0, 2, 1)

        att = torch.bmm(t, p)

        if self.use_scale:
            att = att.div(c**0.5)

        att = self.softmax(att)
        x = torch.bmm(att, g)

        x = x.permute(0, 2, 1)
        x = x.contiguous()
        x = x.view(b, c, h, w)

        x = self.z(x)
        x = self.bn(x) + residual

        return [x]



class FDM(nn.Module):
    def __init__(self, inplanes, planes, use_scale=False, sigma = 2, norm = False):
        self.use_scale = use_scale
        super(FDM, self).__init__()
        self.gate_head_pos = SepConvHead(1,inplanes,inplanes)
        #self.gate_head_neg = SepConvHead(1,inplanes,inplanes)
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(inplanes)
        self.sigma = sigma
        self.norm = norm
        self.z2 = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes)

    def forward(self, x, click_maps):
        #print(x.device, weight_map.device, weight_map.shape)
        residual = x
        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, h, w = t.size()
        t = t.view(b, c, -1).permute(0, 2, 1) # b, hw, c
        p = p.view(b, c, -1)                  # b, c, hw
        g = g.view(b, c, -1).permute(0, 2, 1) # b, hw, c

        gate_map_pos_ori = F.sigmoid(self.gate_head_pos(x))
        gate_map_neg_ori = 1 - gate_map_pos_ori

        gate_map_pos = gate_map_pos_ori.clone().detach() * 0.9 + 0.1
        gate_map_neg = gate_map_neg_ori.clone().detach() * 0.9 + 0.1

        weight_map_pos = click_maps[:,0,:,:].unsqueeze(1)*0.9 + 0.1  #b,1,h,w
        weight_map_neg = click_maps[:,1,:,:].unsqueeze(1)*0.9 + 0.1
        weight_map_pos = weight_map_pos.clone().detach()
        weight_map_neg = weight_map_neg.clone().detach()

        att = torch.bmm(t, p) #b, hw, hw
        att = att.view(b,h*w, h, w) #b,hw,h,w

        att_pos = att *  weight_map_pos
        att_pos = att_pos.view(b,h*w,h*w)
        att_pos = self.softmax(att_pos)
        att_pos = att_pos.view(b,h,w,h*w)
        att_pos = att_pos * gate_map_pos.permute(0,2,3,1)
        att_pos = att_pos.view(b,w*h,w*h)
        x_pos = torch.bmm(att_pos, g)
        x_pos = x_pos.permute(0, 2, 1)
        x_pos = x_pos.contiguous()
        x_pos = x_pos.view(b, c, h, w)
        x_pos = self.z(x_pos)

        att_neg = att *  weight_map_neg
        att_neg = att_neg.view(b,h*w,h*w)
        att_neg = self.softmax(att_neg)
        att_neg = att_neg.view(b,h,w,h*w)
        att_neg = att_neg * gate_map_neg.permute(0,2,3,1)
        att_neg = att_neg.view(b,w*h,w*h)
        x_neg = torch.bmm(att_neg, g)
        x_neg = x_neg.permute(0, 2, 1)
        x_neg = x_neg.contiguous()
        x_neg = x_neg.view(b, c, h, w)
        x_neg = self.z2(x_neg)

        x = self.bn(x_pos) + residual + self.bn2(x_neg)
        return [x, gate_map_pos_ori, gate_map_neg_ori]



class FDM_v2(nn.Module):
    def __init__(self, inplanes, planes, use_scale=False, sigma = 2, norm = False):
        self.use_scale = use_scale
        super(FDM_v2, self).__init__()
        self.gate_head_pos = SepConvHead(1,inplanes,inplanes)
        #self.gate_head_neg = SepConvHead(1,inplanes,inplanes)
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(inplanes)
        self.sigma = sigma
        self.norm = norm
        self.z2 = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes)

    def forward(self, x, click_maps):
        #print(x.device, weight_map.device, weight_map.shape)
        residual = x
        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, h, w = t.size()
        t = t.view(b, c, -1).permute(0, 2, 1) # b, hw, c
        p = p.view(b, c, -1)                  # b, c, hw
        g = g.view(b, c, -1).permute(0, 2, 1) # b, hw, c

        gate_map_pos_ori = F.sigmoid(self.gate_head_pos(x))
        gate_map_neg_ori = 1 - gate_map_pos_ori

        gate_map_pos = gate_map_pos_ori.clone().detach()
        gate_map_neg = gate_map_neg_ori.clone().detach()

        weight_map_pos = click_maps[:,0,:,:].unsqueeze(1)
        weight_map_neg = click_maps[:,1,:,:].unsqueeze(1) * 0.5 + gate_map_neg * 0.5

        weight_map_pos = weight_map_pos.clone().detach()
        weight_map_neg = weight_map_neg.clone().detach()

        att = torch.bmm(t, p) #b, hw, hw
        att = att.view(b,h*w, h, w) #b,hw,h,w

        att_pos = att *  weight_map_pos
        att_pos = att_pos.view(b,h*w,h*w)
        att_pos = self.softmax(att_pos)
        att_pos = att_pos.view(b,h,w,h*w)
        att_pos = att_pos * gate_map_pos.permute(0,2,3,1)
        att_pos = att_pos.view(b,w*h,w*h)
        x_pos = torch.bmm(att_pos, g)
        x_pos = x_pos.permute(0, 2, 1)
        x_pos = x_pos.contiguous()
        x_pos = x_pos.view(b, c, h, w)
        x_pos = self.z(x_pos)

        att_neg = att *  weight_map_neg
        att_neg = att_neg.view(b,h*w,h*w)
        att_neg = self.softmax(att_neg)
        att_neg = att_neg.view(b,h,w,h*w)
        att_neg = att_neg * gate_map_neg.permute(0,2,3,1)
        att_neg = att_neg.view(b,w*h,w*h)
        x_neg = torch.bmm(att_neg, g)
        x_neg = x_neg.permute(0, 2, 1)
        x_neg = x_neg.contiguous()
        x_neg = x_neg.view(b, c, h, w)
        x_neg = self.z2(x_neg)

        x = self.bn(x_pos) + residual + self.bn2(x_neg)
        return [x, gate_map_pos_ori, gate_map_neg_ori]

 

 
class FDM_v3(nn.Module):
    def __init__(self, inplanes, planes, use_scale=False, sigma = 2, norm = False):
        self.use_scale = use_scale
        super(FDM_v3, self).__init__()
        self.gate_head_pos = SepConvHead(1,inplanes,inplanes)
        #self.gate_head_neg = SepConvHead(1,inplanes,inplanes)
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(inplanes)
        self.sigma = sigma
        self.norm = norm
        self.z2 = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes)

    def forward(self, x, click_maps):
        #print(x.device, weight_map.device, weight_map.shape)
        residual = x
        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, h, w = t.size()
        t = t.view(b, c, -1).permute(0, 2, 1) # b, hw, c
        p = p.view(b, c, -1)                  # b, c, hw
        g = g.view(b, c, -1).permute(0, 2, 1) # b, hw, c

        gate_map_pos_ori = F.sigmoid(self.gate_head_pos(x))
        gate_map_neg_ori = 1 - gate_map_pos_ori

        gate_map_pos = gate_map_pos_ori.clone().detach()
        gate_map_neg = gate_map_neg_ori.clone().detach()

        weight_map_pos = click_maps[:,0,:,:].unsqueeze(1) + (gate_map_pos > 0.8) * 1.0 
        weight_map_neg = click_maps[:,1,:,:].unsqueeze(1) + (gate_map_neg > 0.8) * 1.0 

        weight_map_pos = weight_map_pos.clone().detach()
        weight_map_neg = weight_map_neg.clone().detach()

        att = torch.bmm(t, p) #b, hw, hw
        att = att.view(b,h*w, h, w) #b,hw,h,w

        att_pos = att *  weight_map_pos
        att_pos = att_pos.view(b,h*w,h*w)
        att_pos = self.softmax(att_pos)
        att_pos = att_pos.view(b,h,w,h*w)
        att_pos = att_pos * gate_map_pos.permute(0,2,3,1)
        att_pos = att_pos.view(b,w*h,w*h)
        x_pos = torch.bmm(att_pos, g)
        x_pos = x_pos.permute(0, 2, 1)
        x_pos = x_pos.contiguous()
        x_pos = x_pos.view(b, c, h, w)
        x_pos = self.z(x_pos)

        att_neg = att *  weight_map_neg
        att_neg = att_neg.view(b,h*w,h*w)
        att_neg = self.softmax(att_neg)
        att_neg = att_neg.view(b,h,w,h*w)
        att_neg = att_neg * gate_map_neg.permute(0,2,3,1)
        att_neg = att_neg.view(b,w*h,w*h)
        x_neg = torch.bmm(att_neg, g)
        x_neg = x_neg.permute(0, 2, 1)
        x_neg = x_neg.contiguous()
        x_neg = x_neg.view(b, c, h, w)
        x_neg = self.z2(x_neg)

        x = self.bn(x_pos) + residual + self.bn2(x_neg)
        return [x, gate_map_pos_ori, gate_map_neg_ori]