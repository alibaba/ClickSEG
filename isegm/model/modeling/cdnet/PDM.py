'''
The code of local affinity calculation is adopted from  https://github.com/visinf/1-stage-wseg
'''
import torch
import torch.nn.functional as F
import torch.nn as nn



class LocalAffinity(nn.Module):

    def __init__(self, dilations=[1]):
        super(LocalAffinity, self).__init__()
        self.dilations = dilations
        weight = self._init_aff()
        self.register_buffer('kernel', weight)
        self.distance_decay = 1.5

        decay_list = [1]
        decay_weight = 1
        for i in range(len(self.dilations)-1 ):
            decay_weight = decay_weight * self.distance_decay
            decay_list.append(decay_weight)
        self.deccay_list = decay_list


    def _init_aff(self):
        # initialising the shift kernel
        weight = torch.zeros(8, 1, 3, 3)

        for i in range(weight.size(0)):
            weight[i, 0, 1, 1] = 1

        weight[0, 0, 0, 0] = -1
        weight[1, 0, 0, 1] = -1
        weight[2, 0, 0, 2] = -1

        weight[3, 0, 1, 0] = -1
        weight[4, 0, 1, 2] = -1

        weight[5, 0, 2, 0] = -1
        weight[6, 0, 2, 1] = -1
        weight[7, 0, 2, 2] = -1

        self.weight_check = weight.clone()

        return weight

    def forward(self, x):
        self.weight_check = self.weight_check.type_as(x)
        assert torch.all(self.weight_check.eq(self.kernel))

        B,K,H,W = x.size()
        x = x.view(B*K,1,H,W)

        x_affs = []
        for i in range(len(self.dilations)):
            d = self.dilations[i]
            x_pad = F.pad(x, [d]*4, mode='replicate')
            x_aff = F.conv2d(x_pad, self.kernel, dilation=d) #* self.deccay_list[i]
            x_affs.append(x_aff)

        x_aff = torch.cat(x_affs, 1)
        return x_aff.view(B,K,-1,H,W)

class LocalAffinityCopy(LocalAffinity):

    def _init_aff(self):
        # initialising the shift kernel
        weight = torch.zeros(8, 1, 3, 3)

        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 2] = 1

        weight[5, 0, 2, 0] = 1
        weight[6, 0, 2, 1] = 1
        weight[7, 0, 2, 2] = 1

        self.weight_check = weight.clone()
        return weight

class LocalStDev(LocalAffinity):

    def _init_aff(self):
        weight = torch.zeros(9, 1, 3, 3)
        weight.zero_()

        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 1] = 1
        weight[5, 0, 1, 2] = 1

        weight[6, 0, 2, 0] = 1
        weight[7, 0, 2, 1] = 1
        weight[8, 0, 2, 2] = 1

        self.weight_check = weight.clone()
        return weight

    def forward(self, x):
        # returns (B,K,P,H,W), where P is the number
        # of locations
        x = super(LocalStDev, self).forward(x)
        #print(x.shape) #1,3,36,480,854
        return x.std(2, keepdim=True)

class LocalAffinityAbs(LocalAffinity):
    def forward(self, x):
        x = super(LocalAffinityAbs, self).forward(x)
        #return torch.pow(x,2)
        return torch.abs(x)

#
# Pixel Diffusion Module
#
class PDM(nn.Module):
    def __init__(self, num_iter=5, dilations=[1,2,4,8]):
        super(PDM, self).__init__()

        self.num_iter = num_iter
        self.aff_x = LocalAffinityAbs(dilations)
        self.aff_m = LocalAffinityCopy(dilations)
        self.aff_std = LocalStDev(dilations)

    def forward(self, image, mask, gaussian_maps=None ):
        x = image
        mask = F.interpolate(mask, size=x.size()[-2:], mode="bilinear", align_corners=True)
        mask = (torch.sigmoid(mask)-0.5) * 2
        mask_residual = mask

        fil = torch.abs(mask) > 0.25
        mask = mask * fil
        # x: [BxKxHxW]
        # mask: [BxCxHxW]
        B,K,H,W = x.size()
        _,C,_,_ = mask.size()

        x_std = self.aff_std(x)
        x = -self.aff_x(x) / (1e-8 + 0.1 * x_std)
        x = x.mean(1, keepdim=True)
        thresh = x.mean()
        fil = x > thresh
        x = F.softmax(x, 2)

        if gaussian_maps != None:
            pos_map = gaussian_maps[:,0,:,:]
            neg_map = gaussian_maps[:,1,:,:]
            gaussian_maps = torch.max(gaussian_maps,1)[0].unsqueeze(1)#1,1,480,854

        amp = 10
        mask = mask  + pos_map * amp  - neg_map * amp
        for _ in range(self.num_iter):
            m = self.aff_m(mask)  # [BxCxPxHxW]
            mask = (m * x).sum(2)
        mask = mask + mask_residual
        # xvals: [BxCxHxW]
        #print(mask.max(), mask.min())
        return mask


