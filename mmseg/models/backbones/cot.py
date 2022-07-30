import numpy as np
import torch
import torch.nn as nn
from functools import partial
import math
from itertools import repeat
import matplotlib.pyplot as plt

import torchvision.transforms
from torch._six import container_abcs
import warnings
from .. import builder
import time
import torch.nn.functional as F

import sys

# sys.path.append("../..")
# from tools.contourlet_transform.contourlet_dec import ContourletDec
from tools.contourlet_transform.tools.shownsct import shownsct

from .helpers import load_pretrained
# from .layers import DropPath, to_2tuple, trunc_normal_

from ..builder import BACKBONES

from mmcv.cnn import build_norm_layer


@BACKBONES.register_module()
class ContourletTransformer_l3(nn.Module):
    def __init__(self, norm_cfg, vit_backbone, resnet_backbone_1, resnet_backbone_2, resnet_backbone_3, pretrained, ms_nsct=False):
        super(VIT_NSCT, self).__init__()
        self.vit_backbone = builder.build_backbone(vit_backbone)
        self.resnet_backbone_1 = builder.build_backbone(resnet_backbone_1)
        self.resnet_backbone_2 = builder.build_backbone(resnet_backbone_2)
        self.resnet_backbone_3 = builder.build_backbone(resnet_backbone_3)

        self.ms_nsct = ms_nsct
        # self.sobel_x, self.sobel_y = self._init_sobel_filter()
        _, self.x2_syncbn = build_norm_layer(norm_cfg, 1024)
        _, self.nsct_syncbn = build_norm_layer(norm_cfg, 7)

        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        # if not isinstance(pretrained, dict):
        #     ValueError('Need two pre trained address')
        if pretrained is None:
            pretrained = {'vit_pretrained': None,
                          'resnet_pretrained': None}
        self.vit_backbone.init_weights(pretrained['vit_pretrained'])
        self.resnet_backbone_1.init_weights(pretrained['resnet_pretrained'])
        self.resnet_backbone_2.init_weights(pretrained['resnet_pretrained'])
        self.resnet_backbone_3.init_weights(pretrained['resnet_pretrained'])

    def _init_sobel_filter(self):
        Kv = np.asarray([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        Kh = np.asarray([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        return torch.from_numpy(Kv).unsqueeze(0).unsqueeze(0).float().cuda(), \
               torch.from_numpy(Kh).unsqueeze(0).unsqueeze(0).float().cuda()

    def sobel_dec(self, x):
        gray_image = torchvision.transforms.Grayscale()(x)
        x_image = torch.nn.functional.conv2d(gray_image, self.sobel_x, stride=1, padding=1)
        y_image = torch.nn.functional.conv2d(gray_image, self.sobel_y, stride=1, padding=1)
        z_image = y_image.clone()
        sobel_image = torch.cat((x_image, y_image, z_image), dim=1)

        # x_image_np = x_image[0].squeeze(0).cpu().numpy()
        # y_image_np = y_image[0].squeeze(0).cpu().numpy()
        # plt.subplot(1, 2, 1)
        # plt.imshow(x_image_np, cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(y_image_np, cmap='gray')
        # plt.show()
        return sobel_image

    def nsct_norm(self, x2):
        x2_feature = x2[1]
        for feat in x2[2:]:
            if isinstance(feat, list):
                for dir in feat:
                    x2_feature = torch.cat((x2_feature, dir), dim=1)
            else:
                x2_feature = torch.cat((x2_feature, feat), dim=1)
        x2_feature = self.nsct_syncbn(x2_feature)

        return x2_feature

    def forward(self, x):

        # import matplotlib.pyplot as plt
        # x_np = x.cpu().numpy()
        # for i in range(x_np.shape[1]):
        #     plt.imshow(x_np[2, i, :, :], cmap='gray')
        #     plt.title('level{}'.format(i))
        #     plt.show()

        # x2 = torchvision.transforms.Grayscale()(x)

        # x2 = self.nsct.dec_iter(x2)

        # shownsct(x2, gpu=True)
        low_freq = x[:, 3: 4]
        x2 = x[:, 4:]
        x2 = self.nsct_syncbn(x2)
        # x2 = self.nsct_norm(x2)

        x2_1 = x2[:, 0: 1]
        x2_2 = x2[:, 1: 3]
        x2_3 = x2[:, 3: 7]

        if self.ms_nsct:
            x2_2 = F.interpolate(x2_2, x2_3.shape[-1] // 2, mode='bilinear', align_corners=True)
            x2_1 = F.interpolate(x2_1, x2_3.shape[-1] // 4, mode='bilinear', align_corners=True)
            low_freq = F.interpolate(low_freq, x2_3.shape[-1] // 8, mode='bilinear', align_corners=True)


        nsct_features = [self.resnet_backbone_1(x2_1)[0],
                         self.resnet_backbone_2(x2_2)[0],
                         self.resnet_backbone_3(x2_3)[0]]

        tf_feature, nsct_1, nsct_2, nsct_3 = self.vit_backbone(x[:, 0: 3],
                                                               nsct_features=nsct_features,
                                                               low_freq=low_freq)

        return tf_feature, nsct_1, nsct_2, nsct_3


@BACKBONES.register_module()
class ContourletTransformer(nn.Module):
    def __init__(self, norm_cfg, vit_backbone, resnet_backbone_1, resnet_backbone_2, pretrained, ms_nsct=False):
        super(ContourletTransformer, self).__init__()

        self.pretrained = pretrained
        vit_backbone.pretrained = self.pretrained['vit_pretrained']

        self.vit_backbone = builder.build_backbone(vit_backbone)
        self.resnet_backbone_1 = builder.build_backbone(resnet_backbone_1)
        self.resnet_backbone_2 = builder.build_backbone(resnet_backbone_2)

        self.ms_nsct = ms_nsct
        # self.sobel_x, self.sobel_y = self._init_sobel_filter()
        _, self.x2_syncbn = build_norm_layer(norm_cfg, 1024)
        _, self.nsct_syncbn = build_norm_layer(norm_cfg, 3)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        # if not isinstance(pretrained, dict):
        #     ValueError('Need two pre trained address')
        if pretrained is None:
            pretrained = {'vit_pretrained': None,
                          'resnet_pretrained': None}
        # self.vit_backbone.init_weights(pretrained['vit_pretrained'])
        self.vit_backbone.init_weights()
        self.resnet_backbone_1.init_weights(pretrained['resnet_pretrained'])
        self.resnet_backbone_2.init_weights(pretrained['resnet_pretrained'])

    def forward(self, x):

        x1_low = x[:, 3: 4]
        x_dirs = x[:, 4:]
        x_dirs = self.nsct_syncbn(x_dirs)
        # x2 = self.nsct_norm(x2)

        x1_dirs = x_dirs[:, 0: 1]
        x2_dirs = x_dirs[:, 1: 3]

        if self.ms_nsct:
            x1_dirs = F.interpolate(x1_dirs, x2_dirs.shape[-1] // 2, mode='bilinear', align_corners=True)
            x1_low = F.interpolate(x1_low, x2_dirs.shape[-1] // 4, mode='bilinear', align_corners=True)

        hf_features = (self.resnet_backbone_1(x1_dirs)[0], self.resnet_backbone_2(x2_dirs)[0])  # high frequency features

        vit_outs = self.vit_backbone(x[:, 0: 3])

        return vit_outs, hf_features  # return a tuple
