import os.path
import sys

import numpy
import numpy as np
import torch.nn as nn
import torch
import copy
import mmcv
from mmcv.cnn import ConvModule, build_norm_layer
from ..decode_heads.decode_head import BaseDecodeHead
from ..decode_heads.psp_head import PPM
from mmseg.ops import resize

from ..builder import NECKS
from .. import builder

import torch.nn.functional as F
import math

def vis_hidden_layer(bs_feature, img_metas):
    import matplotlib.pyplot as plt
    bs = bs_feature.shape[0]
    for n, feature in enumerate(bs_feature):
        layer_for_vis = torch.sum(feature, dim=0, keepdim=False)
        # layer_for_vis = feature[0]

        # norm
        layer_for_vis = layer_for_vis.detach().cpu().numpy()
        layer_for_vis = (layer_for_vis - numpy.mean(layer_for_vis)) / numpy.std(layer_for_vis)
        layer_for_vis = (layer_for_vis - layer_for_vis.min()) / (layer_for_vis.max() - layer_for_vis.min())


        # layer_for_vis[layer_for_vis > (numpy.mean(layer_for_vis) + 0.2)] = numpy.mean(layer_for_vis)

        plt.imshow(layer_for_vis)

        os.makedirs('hidden_layer_result_images', exist_ok=True)
        save_pth = os.path.join('hidden_layer_result_images', img_metas[n]['ori_filename'])

        plt.axis('off')
        plt.savefig(save_pth, dpi=600)
        # plt.show()
        plt.close()

@NECKS.register_module()
class CoTNeckV2(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, deficient_ratio, ct_levels, dfb_stage, sparse_resnet_config, pool_scales=(1, 2, 3, 6), **kwargs):
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

        self.dfb_stage = dfb_stage
        assert ct_levels == len(dfb_stage)
        for i in range(self.ct_levels):
            sparse_resnet_config['in_channels'] = 2 ** dfb_stage[i]
            # if i == 0:
            #     assert sparse_resnet_config.get('in_channels') == 1
            # elif i == 1:
            #     assert sparse_resnet_config.get('in_channels') == 1
            #     sparse_resnet_config['in_channels'] = 2
            # elif i == 2:
            #     assert sparse_resnet_config.get('in_channels') == 2
            #     sparse_resnet_config['in_channels'] = 4
            # else:
            #     raise ValueError('ct_levels extend 3')

            self.sparse_resnet.append(builder.build_backbone(sparse_resnet_config))
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

        if sparse_resnet_config.get('num_stages') == 1:
            self.dir_feat_channel_mapping = ConvModule(
                256,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
        self.is_vanilla_resnet = True if sparse_resnet_config.get('type') == 'ResNet' else False

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

    def _get_pseudo_gt(self, output):
        _, pseudo_gt = output.topk(1, dim=1)

        return pseudo_gt.long()

    def _find_deficient_points(self, vanilla_output, gt):

        if not self.training:
            assert gt is None
            gt = self._get_pseudo_gt(vanilla_output)

        gt = resize(gt.float(), vanilla_output.shape[-2:]).long()

        bs = vanilla_output.shape[0]

        mask_indexes = torch.where(gt.squeeze(1) == 255)

        gt = F.one_hot(gt.squeeze(1), num_classes=256)[..., :self.num_classes]
        gt = gt.permute(0, 3, 1, 2)

        confidence_map = torch.sum(gt * vanilla_output, dim=1)
        confidence_map[mask_indexes] = 999

        n_points = int(vanilla_output.shape[-1] * vanilla_output.shape[-2] * self.deficient_ratio)
        _, deficient_coords = confidence_map.reshape(bs, -1).topk(n_points, largest=False)

        deficient_map = torch.zeros_like(confidence_map)

        for i in range(bs):
            deficient_map[i].reshape(-1)[deficient_coords[i]] = 1.0

        return deficient_map.unsqueeze(1)

    def _forward_dirs_feat(self, deficient_map, ct_dirs, level):
        if self.is_vanilla_resnet:
            dir_feat = self.sparse_resnet[level](deficient_map * ct_dirs)[0]
        else:
            dir_feat = self.sparse_resnet[level](deficient_map, ct_dirs)[0]
        if hasattr(self, 'dir_feat_channel_mapping'):
            dir_feat = self.dir_feat_channel_mapping(dir_feat)
        return dir_feat


    def _cot_upsample(self, deep_feats, ct_dirs, shallow_feats, deficient_map, level, img_metas=None):
        if deficient_map.shape[-2:] != shallow_feats.shape[-2:]:
            deficient_map = resize(deficient_map,
                                   shallow_feats.shape[-2:],
                                   mode='nearest')

        if ct_dirs.shape[-2:] != shallow_feats.shape[-2:]:
            ct_dirs = resize(ct_dirs,
                             shallow_feats.shape[-2:],
                             mode='bilinear',
                             align_corners=self.align_corners)

        # sparse_dirs = deficient_map * ct_dirs
        dir_feats = self._forward_dirs_feat(deficient_map, ct_dirs, level)
        #
        # if level == 0:
        #     vis_hidden_layer(dir_feats, img_metas)

        # fuse direction feature with deep feature,
        # and upsample 2x

        recomposed_feats = shallow_feats + dir_feats + resize(deep_feats,
                                                              shallow_feats.shape[-2:],
                                                              mode='bilinear',
                                                              align_corners=self.align_corners)

        recomposed_feats = self.cot_recompose_conv[level](recomposed_feats)

        return recomposed_feats

    def _show_deficient_points(self, deficient_points_maps, imgs, img_metas):
        import matplotlib.pyplot as plt

        bs = deficient_points_maps.shape[0]
        for n in range(bs):
            deficient_points_map = deficient_points_maps[n].transpose(1, 2, 0).squeeze(2)
            decifient_point_coords = np.where(deficient_points_map == 1)

            img = mmcv.imdenormalize(imgs[n].transpose(1, 2, 0), img_metas[n]['img_norm_cfg']['mean'],
                                     img_metas[n]['img_norm_cfg']['std'], to_bgr=False)
            # plt.imshow(img.astype(np.int))
            # plt.show()

            plt.imshow(img.astype(np.int))
            plt.scatter(decifient_point_coords[1], decifient_point_coords[0], s=1.0, marker='o', c='yellow', alpha=0.6)
            os.makedirs('result_images/', exist_ok=True)
            # print(img_metas[n]['ori_filename'])
            save_pth = os.path.join('result_images', img_metas[n]['ori_filename'])
            plt.savefig(save_pth, dpi=600)
            plt.close()

            plt.imshow(img.astype(np.int), alpha=0)
            plt.scatter(decifient_point_coords[1], decifient_point_coords[0], s=1.0, marker='o', c='yellow', alpha=0.8)
            save_ulps_kpth = save_pth.split('.jpg')[0] + '_scatter_only.jpg'
            plt.axis('off')
            plt.savefig(save_ulps_kpth, dpi=600)
            plt.close()

            # plt.show()


    def _forward_feature(self, inputs, x_dirs, gt, img=None, img_metas=None, show_decicient_points=False):
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
        elif self.dfb_stage == [2, 1, 0]:
            ct_dirs = [x_dirs[:, 3: 7], x_dirs[:, 1: 3], x_dirs[:, 0: 1]]
        elif self.dfb_stage == [0, 1, 2]:
            ct_dirs = [x_dirs[:, 6: 7], x_dirs[:, 4: 6], x_dirs[:, 0: 4]]
        elif self.dfb_stage == [0, 0, 0]:
            ct_dirs = [x_dirs[:, 2: 3], x_dirs[:, 1: 2], x_dirs[:, 0: 1]]
        elif self.dfb_stage == [2, 2, 2]:
            ct_dirs = [x_dirs[:, 8: 12], x_dirs[:, 4: 8], x_dirs[:, 0: 4]]
        else:
            raise ValueError('ct_levels should be 2 or 3')

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

        if show_decicient_points:
            img = resize(img, size=deficient_points_map.shape[-2:], mode='bilinear', align_corners=self.align_corners)
            self._show_deficient_points(deficient_points_map.detach().cpu().numpy(),
                                        img.detach().cpu().numpy(), img_metas)
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
                # try:
                cot_recompose[i - 1] = self._cot_upsample(cot_recompose[i],
                                                          ct_dirs[i - 1],
                                                          cot_recompose[i - 1],
                                                          deficient_points_map,
                                                          i - 1,
                                                          img_metas)
                # except:
                #     img = resize(img, size=deficient_points_map.shape[-2:], mode='bilinear', align_corners=self.align_corners)
                #     self._show_deficient_points(deficient_points_map.detach().cpu().numpy(),
                #                                 img.detach().cpu().numpy(), img_metas)
                #     for img_meta in img_metas:
                #         print(img_meta)
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

    def forward(self, inputs, img, gt_semantic_seg=None, img_metas=None, show_decicient_points=False):
        """Forward function."""
        x_dirs = img[:, 4:]

        #  check input channels
        dfb_channels = 0
        for i in self.dfb_stage:
            dfb_channels += 2**i
        assert x_dirs.shape[1] == dfb_channels

        if show_decicient_points:
            assert img_metas is not None
            cot_feats, cot_mid_feats, vanilla_output = self._forward_feature(inputs, x_dirs, gt_semantic_seg,
                                                                             img[:, 0:3], img_metas,
                                                                             show_decicient_points=True)
        else:
            cot_feats, cot_mid_feats, vanilla_output = self._forward_feature(inputs, x_dirs, gt_semantic_seg)

        assert len(cot_mid_feats) == 3
        cot_mid_feats_1 = cot_mid_feats[0]
        cot_mid_feats_2 = cot_mid_feats[1]
        cot_mid_feats_3 = cot_mid_feats[2]

        return cot_feats, \
               cot_mid_feats_1, cot_mid_feats_2, cot_mid_feats_3, \
               vanilla_output

