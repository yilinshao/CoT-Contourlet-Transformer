import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer

from ..builder import NECKS
from .. import builder

import torch.nn.functional as F
import math

class NsctFusion01(nn.Module):
    def __init__(self, in_channels=(1024, 1024, 512, 512), mla_channels=256, norm_cfg=None):
        super(NsctFusion01, self).__init__()
        self.tf_1x1_c18 = nn.Sequential(nn.Conv2d(
            in_channels[0], mla_channels, 1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.tf_1x1 = nn.Sequential(nn.Conv2d(
            in_channels[1], mla_channels, 1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.nsct_1_1x1 = nn.Sequential(nn.Conv2d(
            in_channels[2], mla_channels, 1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())
        self.nsct_2_1x1 = nn.Sequential(nn.Conv2d(
            in_channels[3], mla_channels, 1, bias=False), build_norm_layer(norm_cfg, mla_channels)[1], nn.ReLU())

        self.nsct_1 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                     build_norm_layer(norm_cfg, mla_channels)[1],
                                     nn.ReLU(),
                                     nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                     build_norm_layer(norm_cfg, mla_channels)[1],
                                     nn.ReLU(),
                                     nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                     build_norm_layer(norm_cfg, mla_channels)[1],
                                     nn.ReLU())
        self.nsct_2 = nn.Sequential(nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                     build_norm_layer(norm_cfg, mla_channels)[1],
                                     nn.ReLU(),
                                     nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                     build_norm_layer(norm_cfg, mla_channels)[1],
                                     nn.ReLU(),
                                     nn.Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                                     build_norm_layer(norm_cfg, mla_channels)[1],
                                     nn.ReLU())

    def to_2D(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward(self, tf_feature_c18, tf_feature_c24, nsct_1, nsct_2):

        tf_feature_c18 = self.tf_1x1_c18(tf_feature_c18)

        tf_feature = F.interpolate(tf_feature_c24, nsct_1.shape[-1], mode='bilinear', align_corners=True)
        tf_feature = self.tf_1x1(tf_feature)
        nsct_1_1x1 = self.nsct_1_1x1(nsct_1)
        nsct_1_plus = tf_feature + nsct_1_1x1
        nsct_1_plus = self.nsct_1(nsct_1_plus)

        nsct_1_plus = F.interpolate(nsct_1_plus, nsct_2.shape[-1], mode='bilinear', align_corners=True)
        nsct_2_1x1 = self.nsct_2_1x1(nsct_2)
        nsct_2_plus = nsct_1_plus + nsct_2_1x1
        nsct_2_plus = self.nsct_2(nsct_2_plus)

        return tf_feature_c18, tf_feature, nsct_1_plus, nsct_2_plus


@NECKS.register_module()
class CoTNeck(nn.Module):
    """Multi-level Feature Aggregation.

    This neck is `The Multi-level Feature Aggregation construction of
    SETR <https://arxiv.org/abs/2012.15840>`_.


    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        norm_layer (dict): Config dict for input normalization.
            Default: norm_layer=dict(type='LN', eps=1e-6, requires_grad=True).
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 setr_mla_neck,
                 norm_layer=dict(type='LN', eps=1e-6, requires_grad=True),
                 norm_cfg=None,
                 act_cfg=None):
        super(CoTNeck, self).__init__()

        self.setr_mla_neck = builder.build_neck(setr_mla_neck)

        self.nsct_fusion = NsctFusion01(in_channels=(setr_mla_neck.out_channels, setr_mla_neck.out_channels, 512, 512),
                                        mla_channels=setr_mla_neck.out_channels,
                                        norm_cfg=norm_cfg)


    def forward(self, inputs):
        fuse_vit_outs = self.setr_mla_neck(inputs[0])

        tf_feature_c18, tf_feature_c24, nsct_1, nsct_2 = self.nsct_fusion(fuse_vit_outs[-2],
                                                                          fuse_vit_outs[-1],
                                                                          inputs[1][0],
                                                                          inputs[1][1])

        return (tf_feature_c18, tf_feature_c24, nsct_1, nsct_2)

