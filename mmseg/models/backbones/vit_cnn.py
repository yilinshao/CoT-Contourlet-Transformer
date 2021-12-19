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

from .helpers import load_pretrained
# from .layers import DropPath, to_2tuple, trunc_normal_

from ..builder import BACKBONES

from mmcv.cnn import build_norm_layer


@BACKBONES.register_module()
class VIT_CNN(nn.Module):
    def __init__(self, norm_cfg, vit_backbone, resnet_backbone):
        super(VIT_CNN, self).__init__()
        self.vit_backbone = builder.build_backbone(vit_backbone)
        self.resnet_backbone = builder.build_backbone(resnet_backbone)
        self.sobel_x, self.sobel_y = self._init_sobel_filter()
        _, self.x2_syncbn = build_norm_layer(norm_cfg, 1024)

    def init_weights(self, pretrained=None):
        # if not isinstance(pretrained, dict):
        #     ValueError('Need two pre trained address')
        if pretrained is None:
            pretrained = {'vit_pretrained': None,
                          'resnet_pretrained': None}
        self.vit_backbone.init_weights(pretrained['vit_pretrained'])
        self.resnet_backbone.init_weights(pretrained['resnet_pretrained'])

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

    def forward(self, x):
        x1 = self.vit_backbone(x)

        x2 = self.sobel_dec(x)
        x2 = self.resnet_backbone(x2)

        x2 = x2[0]
        x2 = self.x2_syncbn(x2)

        x2_out = torch.zeros_like(x1[self.vit_backbone.out_indices[0]])

        x2 = x2.reshape((x2_out.size(0), x2_out.size(2), -1)).transpose(1, 2)
        x2_out[:, 1:] = x2

        x1 = [None if x1_item is None else torch.cat([x2_out, x1_item], dim=2) for x1_item in x1]

        return x1




