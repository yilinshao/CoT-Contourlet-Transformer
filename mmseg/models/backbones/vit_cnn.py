import numpy as np
import torch
import torch.nn as nn
from functools import partial
import math
from itertools import repeat
from PIL import Image

import torchvision.transforms
from torch._six import container_abcs
import warnings
from .. import builder

from .helpers import load_pretrained
# from .layers import DropPath, to_2tuple, trunc_normal_

from ..builder import BACKBONES

from mmcv.cnn import build_norm_layer


@BACKBONES.register_module
class VIT_CNN(nn.Module):
    def __init__(self, vit_backbone, resnet_backbone):
        super(VIT_CNN, self).__init__()
        self.vit_backbone = builder.build_backbone(vit_backbone)
        self.resnet_backbone = builder.build_backbone(resnet_backbone)
        self.sobel_x, self.sobel_y = self._init_sobel_filter()

    def init_weights(self, pretrained=None):
        if not isinstance(pretrained, dict):
            ValueError('Need two pre trained address')
        self.vit_backbone.init_weights(pretrained['vit_pretrained'])
        self.resnet_backbone.init_weights(pretrained['resnet_pretrained'])

    def _init_sobel_filter(self):

        Kv = np.asarray([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        Kh = np.asarray([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        return torch.from_numpy(Kv).unsqueeze(0).unsqueeze(0).cuda(), \
               torch.from_numpy(Kh).unsqueeze(0).unsqueeze(0).cuda()

    def sobel_dec(self, x):
        gray_image = torchvision.transforms.Grayscale()(x)
        x_image = torch.nn.functional.conv2d(gray_image, self.sobel_x, stride=1, padding=1)
        y_image = torch.nn.functional.conv2d(gray_image, self.sobel_y, stride=1, padding=1)
        sobel_image = torch.cat((x_image, y_image), dim=1)

        x_image_np = x_image.cpu().numpy()
        img_tr = Image.fromarray(x_image_np)
        img_tr.show("L")
        return sobel_image

    def forward(self, x):
        x1 = self.vit_backbone(x)

        x2 = self.sobel_dec(x)
        x2 = self.resnet_backbone(x2)

        return x1




