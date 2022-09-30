import torch.nn as nn
import torch
import copy
from mmcv.cnn import ConvModule, build_norm_layer
from ..decode_heads.decode_head import BaseDecodeHead
from ..decode_heads.psp_head import PPM
from mmseg.ops import resize

from ..builder import NECKS
from .. import builder

import torch.nn.functional as F
import math

@NECKS.register_module()
class CoTNeckV2(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, deficient_ratio, ct_levels, sparse_resnet_config, pool_scales=(1, 2, 3, 6), **kwargs):
        super(CoTNeckV2, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.deficient_ratio = deficient_ratio
        self.ct_levels = ct_levels
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.cot_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.sparse_resnet = nn.ModuleList()
        self.cot_recompose_conv = nn.ModuleList()

        for i in range(self.ct_levels):
            if i == 0:
                assert sparse_resnet_config.get('in_channels') == 1
            elif i == 1:
                assert sparse_resnet_config.get('in_channels') == 1
                sparse_resnet_config['in_channels'] = 2
            elif i == 2:
                assert sparse_resnet_config.get('in_channels') == 2
                sparse_resnet_config['in_channels'] = 4
            else:
                raise ValueError('ct_levels extend 3')

            self.sparse_resnet.insert(0, builder.build_backbone(sparse_resnet_config))
            self.cot_recompose_conv.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False))

        self.cot_conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)



    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, img):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs, img, gt_semantic_seg)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def _find_deficient_points(self, vanilla_output, gt):

        gt = resize(gt.float(), vanilla_output.shape[-2:]).long()

        bs = vanilla_output.shape[0]


        gt = F.one_hot(gt.squeeze(1), num_classes=256)[..., :self.num_classes]
        print('=' * 20)
        print(gt.shape)
        print(gt.squeeze(1).shape)
        print(gt.squeeze(1).max())
        print(self.num_classes)
        print('=' * 20)

        gt = gt.permute(0, 3, 1, 2)

        confidence_map = torch.sum(gt * vanilla_output, dim=1)

        print('='*20)
        print(confidence_map.shape)
        print('='*20)

        n_points = int(vanilla_output.shape[-1] * vanilla_output.shape[-2] * self.deficient_ratio)
        _, deficient_coords = confidence_map.reshape(bs, -1).topk(n_points, largest=False)

        deficient_map = torch.zeros_like(confidence_map)

        for i in range(bs):
            deficient_map[i].reshape(-1)[deficient_coords[i]] = 1.0

        return deficient_map.unsqueeze(1)

    def _forward_dirs_feat(self, sparse_dirs, level):
        dir_feat = self.sparse_resnet[level](sparse_dirs)
        return dir_feat[0]


    def _cot_upsample(self, deep_feats, ct_dirs, shallow_feats, deficient_map, level):

        if deficient_map.shape[-1] != shallow_feats.shape[-2:]:
            deficient_map = resize(deficient_map,
                                   shallow_feats.shape[-2:],
                                   mode='bilinear',
                                   align_corners=self.align_corners)

        if ct_dirs.shape[-2:] != shallow_feats.shape[-2:]:
            ct_dirs = resize(ct_dirs,
                             shallow_feats.shape[-2:],
                             mode='bilinear',
                             align_corners=self.align_corners)

        sparse_dirs = deficient_map * ct_dirs
        dir_feats = self._forward_dirs_feat(sparse_dirs, level)

        # fuse direction feature with deep feature,
        # and upsample 2x

        recomposed_feats = shallow_feats + dir_feats + resize(deep_feats,
                                                              shallow_feats.shape[-2],
                                                              mode='bilinear',
                                                              align_corners=self.align_corners)

        recomposed_feats = self.cot_recompose_conv[level](recomposed_feats)

        return recomposed_feats


    def _forward_feature(self, inputs, x_dirs, gt):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # inputs = [(1/4, 96), (1/8, 192), (1/16, 384), (1/32, 768)]
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # laterals = [(1/4, 512), (1/8, 512), (1/16, 512), (1/32, 512)]
        laterals.append(self.psp_forward(inputs))

        # laterals_dirs[(1, 2), (1, 1)]
        if self.ct_levels == 2:
            ct_dirs = [x_dirs[:, 1: 3], x_dirs[:, 0: 1]]
        else:
            raise ValueError('ct_levels should be 2')

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # fpn_outs = [(1/4, 512), (1/8, 512), (1/16, 512), (1/32, 512)]
        # build outputs
        fpn_outs = []
        cot_recompose = []
        for i in range(used_backbone_levels - 1):
            fpn_out = self.fpn_convs[i](laterals[i])
            fpn_outs.append(fpn_out)
            cot_recompose.append(fpn_out)
        # append psp feature
        fpn_outs.append(laterals[-1])
        cot_recompose.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)

        vanilla_output = self.cls_seg(feats)

        # cot upsample
        # deficient_points_map = (1/4)
        deficient_points_map = self._find_deficient_points(vanilla_output.detach(), gt)
        cot_mid_feats = []
        for i in range(used_backbone_levels - 1, 0, -1):
            if len(ct_dirs) < i:
                prev_shape = cot_recompose[i - 1].shape[2:]
                cot_recompose[i - 1] = cot_recompose[i - 1] + resize(
                    cot_recompose[i],
                    size=prev_shape,
                    mode='bilinear',
                    align_corners=self.align_corners)

                cot_mid_feats.append(cot_recompose[i - 1])

            else:
                cot_recompose[i - 1] = self._cot_upsample(cot_recompose[i],
                                                          ct_dirs[i - 1],
                                                          cot_recompose[i - 1],
                                                          deficient_points_map,
                                                          i - 1)
                cot_mid_feats.append(cot_recompose[i - 1])

        # merge all CoT recomposed features
        for i in range(used_backbone_levels - 1, 0, -1):
            cot_recompose[i] = resize(
                cot_recompose[i],
                size=cot_recompose[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        cot_out = torch.cat(cot_recompose, dim=1)
        cot_feats = self.cot_bottleneck(cot_out)
        #  cot_feats = (1/4, 512)
        #  cot_mid_feats = [(1/16, 512), (1/8, 512), (1/4, 512)]
        #  vanilla_output = (1/4, class_num)
        return cot_feats, cot_mid_feats, vanilla_output

    def forward(self, inputs, img, gt_semantic_seg):
        """Forward function."""
        cot_feats, cot_mid_feats, vanilla_output = self._forward_feature(inputs, img[:, 4:], gt_semantic_seg)
        assert len(cot_mid_feats) == 3
        cot_mid_feats_1 = cot_mid_feats[0]
        cot_mid_feats_2 = cot_mid_feats[1]
        cot_mid_feats_3 = cot_mid_feats[2]

        return cot_feats, \
               cot_mid_feats_1, cot_mid_feats_2, cot_mid_feats_3, \
               vanilla_output

