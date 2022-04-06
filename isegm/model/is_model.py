import torch
import torch.nn as nn
import numpy as np

from isegm.model.ops import DistMaps, ScaleLayer, BatchImageNormalize
from isegm.model.modifiers import LRMult


class ISModel(nn.Module):
    def __init__(self, use_rgb_conv=True, feature_stride = 4, with_aux_output=False,
                 norm_radius=260, use_disks=False, cpu_dist_maps=False,
                 clicks_groups=None, with_prev_mask=False, use_leaky_relu=False,
                 binary_prev_mask=False, conv_extend=False, norm_layer=nn.BatchNorm2d,
                 norm_mean_std=([.485, .456, .406], [.229, .224, .225])):
        super().__init__()
        self.with_aux_output = with_aux_output
        self.clicks_groups = clicks_groups
        self.with_prev_mask = with_prev_mask
        self.binary_prev_mask = binary_prev_mask
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])

        self.coord_feature_ch = 2
        if clicks_groups is not None:
            self.coord_feature_ch *= len(clicks_groups)

        if self.with_prev_mask:
            self.coord_feature_ch += 1

        mt_layers = [
                nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=16, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
                ScaleLayer(init_value=0.05, lr_mult=1)
            ]
        self.maps_transform = nn.Sequential(*mt_layers)
        
        self.dist_maps_2 = DistMaps(norm_radius=2, spatial_scale=1.0,
                                      cpu_mode=cpu_dist_maps, use_disks=use_disks)

        self.dist_maps_5 = DistMaps(norm_radius=5, spatial_scale=1.0,
                                      cpu_mode=cpu_dist_maps, use_disks=use_disks)



    def prepare_input(self, image):
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:, :, :]
            image = image[:, :3, :, :]
            if self.binary_prev_mask:
                prev_mask = (prev_mask > 0.5).float()

        image = self.normalization(image)
        return image, prev_mask

    def backbone_forward(self, image, coord_features=None):
        raise NotImplementedError

    def get_coord_features(self, image, prev_mask, points):
        if self.clicks_groups is not None:
            points_groups = split_points_by_order(points, groups=(2,) + (1, ) * (len(self.clicks_groups) - 2) + (-1,))
            coord_features = [dist_map(image, pg) for dist_map, pg in zip(self.dist_maps, points_groups)]
            coord_features = torch.cat(coord_features, dim=1)
        else:
            coord_features = self.dist_maps_2(image, points)

        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)

        return coord_features

    def load_pretrained_weights(self, path_to_weights= ''):    
        state_dict = self.state_dict()
        pretrained_state_dict = torch.load(path_to_weights, map_location='cpu')['state_dict']
        ckpt_keys = set(pretrained_state_dict.keys())
        own_keys = set(state_dict.keys())
        missing_keys = own_keys - ckpt_keys
        unexpected_keys = ckpt_keys - own_keys
        print('Missing Keys: ', missing_keys)
        print('Unexpected Keys: ', unexpected_keys)
        state_dict.update(pretrained_state_dict)
        self.load_state_dict(state_dict, strict= False)
        '''
        if self.inference_mode:
            for param in self.backbone.parameters():
                param.requires_grad = False
        '''


def split_points_by_order(tpoints: torch.Tensor, groups):
    points = tpoints.cpu().numpy()
    num_groups = len(groups)
    bs = points.shape[0]
    num_points = points.shape[1] // 2

    groups = [x if x > 0 else num_points for x in groups]
    group_points = [np.full((bs, 2 * x, 3), -1, dtype=np.float32)
                    for x in groups]

    last_point_indx_group = np.zeros((bs, num_groups, 2), dtype=np.int)
    for group_indx, group_size in enumerate(groups):
        last_point_indx_group[:, group_indx, 1] = group_size

    for bindx in range(bs):
        for pindx in range(2 * num_points):
            point = points[bindx, pindx, :]
            group_id = int(point[2])
            if group_id < 0:
                continue

            is_negative = int(pindx >= num_points)
            if group_id >= num_groups or (group_id == 0 and is_negative):  # disable negative first click
                group_id = num_groups - 1

            new_point_indx = last_point_indx_group[bindx, group_id, is_negative]
            last_point_indx_group[bindx, group_id, is_negative] += 1

            group_points[group_id][bindx, new_point_indx, :] = point

    group_points = [torch.tensor(x, dtype=tpoints.dtype, device=tpoints.device)
                    for x in group_points]

    return group_points
