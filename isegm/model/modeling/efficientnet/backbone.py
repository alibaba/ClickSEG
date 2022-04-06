"""EfficientNet"""
'''
Code from https://github.com/Tramac/Lightweight-Segmentation/blob/master/light/model/base_model/efficientnet.py
'''
import torch.nn as nn



__all__ = ['EfficientNet', 'get_efficientnet', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
           'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']

class _Swish(nn.Module):
    def __init__(self):
        super(_Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

class _Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(_Hswish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.relu6(x + 3.) / 6.


class SEModuleV2(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SEModuleV2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        se_channels = max(1, int(in_channels * se_ratio))
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, se_channels, 1, bias=False),
            _Swish(),
            nn.Conv2d(se_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        out = self.avg_pool(x)
        out = self.fc(out)
        return x * out.expand_as(x)


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio,
                 dilation=1, se_ratio=0.25, drop_connect_rate=0.2, norm_layer=nn.BatchNorm2d, **kwargs):
        super(MBConvBlock, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.drop_connect_rate = drop_connect_rate
        use_se = (se_ratio is not None) and (0 < se_ratio <= 1.)
        if use_se:
            SELayer = SEModuleV2
        else:
            SELayer = Identity

        layers = list()
        inter_channels = int(round(in_channels * expand_ratio))
        if expand_ratio != 1:
            layers.append(_ConvBNHswish(in_channels, inter_channels, 1, norm_layer=norm_layer))
        layers.extend([
            # dw
            _ConvBNHswish(inter_channels, inter_channels, kernel_size, stride, kernel_size // 2 * dilation, dilation,
                          groups=inter_channels, norm_layer=norm_layer),  # check act function
            SELayer(inter_channels, se_ratio),
            # pw-linear
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        ])
        self.conv = nn.Sequential(*layers)

        if drop_connect_rate:
            self.dropout = nn.Dropout2d(drop_connect_rate)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            if self.drop_connect_rate:
                out = self.dropout(out)
            out = x + out
        return out

class _ConvBNHswish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNHswish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.act = _Hswish(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class EfficientNet(nn.Module):
    def __init__(self, width_coe, depth_coe, depth_divisor=8, min_depth=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, num_classes=1000, dilated=True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(EfficientNet, self).__init__()
        self.width_coe = width_coe
        self.depth_coe = depth_coe
        self.depth_divisor = depth_divisor
        self.min_depth = min_depth
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate  # not use

        layer1_setting = [
            # k, c, n, s, t
            [3, 16, 1, 1, 1]]
        layer2_setting = [
            [3, 24, 2, 2, 6]] #4
        layer3_setting = [
            [5, 40, 2, 2, 6],
            [3, 80, 3, 1, 6]] #8
        layer4_setting = [
            [5, 112, 3, 2, 6]]
        layer5_setting = [
            [5, 192, 4, 2, 6]]
        layer6_setting = [
            [3, 320, 1, 1, 6]]

        # building first layer
        self.in_channels = self.round_filter(32, width_coe, depth_divisor, min_depth)
        self.conv1 = _ConvBNHswish(3, self.in_channels, 3, 2, 1, norm_layer=norm_layer)  # diff from origin

        # building MBConvBlocks
        self.layer1 = self._make_layer(MBConvBlock, layer1_setting, norm_layer=norm_layer)
        self.layer2 = self._make_layer(MBConvBlock, layer2_setting, norm_layer=norm_layer)
        self.layer3 = self._make_layer(MBConvBlock, layer3_setting, norm_layer=norm_layer)
        if dilated:
            self.layer4 = self._make_layer(MBConvBlock, layer4_setting, 2, norm_layer)
            self.layer5 = self._make_layer(MBConvBlock, layer5_setting, 2, norm_layer)
        else:
            self.layer4 = self._make_layer(MBConvBlock, layer4_setting, norm_layer=norm_layer)
            self.layer5 = self._make_layer(MBConvBlock, layer5_setting, norm_layer=norm_layer)
        self.layer6 = self._make_layer(MBConvBlock, layer6_setting, norm_layer=norm_layer)

        '''
        # building last several layers
        last_channels = self.round_filter(1280, width_coe, depth_divisor, min_depth)
        
        self.classifier = nn.Sequential(
            _ConvBNHswish(self.in_channels, last_channels, 1, norm_layer=norm_layer),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(self.dropout_rate),
            nn.Conv2d(last_channels, num_classes, 1))
        '''

    def _make_layer(self, block, block_setting, dilation=1, norm_layer=nn.BatchNorm2d):
        layers = list()

        for k, c, n, s, t in block_setting:
            out_channels = self.round_filter(c, self.width_coe, self.depth_divisor, self.min_depth)
            stride = s if dilation == 1 else 1
            layers.append(block(self.in_channels, out_channels, k, stride, t, dilation, norm_layer=norm_layer))
            self.in_channels = out_channels
            for i in range(n - 1):
                layers.append(block(self.in_channels, out_channels, k, 1, t, 1, norm_layer=norm_layer))
                self.in_channels = out_channels
        return nn.Sequential(*layers)

    @classmethod
    def round_filter(cls, filters, width_coe, depth_divisor, min_depth):
        if not width_coe:
            return filters
        filters *= width_coe
        min_depth = min_depth or depth_divisor
        new_filter = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
        if new_filter < 0.9 * filters:  # prevent rounding by more than 10%
            new_filter += depth_divisor
        return int(new_filter)

    def forward(self, x, side_feature):
        x = self.conv1(x)
        x = x + side_feature
        x = self.layer1(x)
        x = self.layer2(x)
        x4 = x
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x8 = x
        return x4, x8


def get_efficientnet(params, pretrained=False, root='~/,torch/models', **kwargs):
    w, d, _, p = params
    model = EfficientNet(w, d, dropout_rate=p, **kwargs)
    if pretrained:
        raise ValueError("Not support pretrained")
    return model


def efficientnet_b0(**kwargs):
    return get_efficientnet([1.0, 1.0, 224, 0.2], **kwargs)


def efficientnet_b1(**kwargs):
    return get_efficientnet([1.0, 1.1, 240, 0.2], **kwargs)


def efficientnet_b2(**kwargs):
    return get_efficientnet([1.1, 1.2, 260, 0.3], **kwargs)


def efficientnet_b3(**kwargs):
    return get_efficientnet([1.2, 1.4, 300, 0.3], **kwargs)


def efficientnet_b4(**kwargs):
    return get_efficientnet([1.4, 1.8, 380, 0.4], **kwargs)


def efficientnet_b5(**kwargs):
    return get_efficientnet([1.6, 2.2, 456, 0.4], **kwargs)


def efficientnet_b6(**kwargs):
    return get_efficientnet([1.8, 2.6, 528, 0.5], **kwargs)


def efficientnet_b7(**kwargs):
    return get_efficientnet([2.0, 3.1, 600, 0.5], **kwargs)


if __name__ == '__main__':
    model = efficientnet_b7()