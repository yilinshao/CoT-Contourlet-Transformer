# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from .fcn_head import FCNHead

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class PseudoHead(FCNHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self, **kwargs):
        super(PseudoHead, self).__init__(**kwargs)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        return inputs
