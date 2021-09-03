norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=dict(
            vit_pretrained=None,
            resnet_pretrained='transfer/resnet50.pth'
        ),
    backbone=dict(
        type='VIT_NSCT',
        norm_cfg=norm_cfg,
        vit_backbone=dict(
            type='VIT_MLA',
            model_name='vit_large_patch16_384',
            img_size=768,
            patch_size=16,
            in_chans=3,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            num_classes=19,
            drop_rate=0.1,
            fuse_nsct=True,
            norm_cfg=norm_cfg,
            pos_embed_interp=True,
            align_corners=False,
            mla_channels=256,
            mla_index=(5, 11, 17, 23)),
        resnet_backbone_1=dict(
            type='ResNet',
            depth=50,
            in_channels=1,
            num_stages=4,
            frozen_stages=[3, 4],
            out_indices=(1, ),
            dilations=(1, 1, 1, 1),
            strides=(1, 1, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True
        ),
        resnet_backbone_2=dict(
            type='ResNet',
            depth=50,
            in_channels=2,
            num_stages=4,
            frozen_stages=[3, 4],
            out_indices=(1, ),
            dilations=(1, 1, 1, 1),
            strides=(1, 1, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True
        ),
        resnet_backbone_3=dict(
            type='ResNet',
            depth=50,
            in_channels=4,
            num_stages=4,
            frozen_stages=[3, 4],
            out_indices=(1, ),
            dilations=(1, 1, 1, 1),
            strides=(1, 1, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True
        )
    ),
    decode_head=dict(
        type='VIT_MLAHead',
        in_channels=1024,
        channels=512,
        img_size=768,
        mla_channels=256,
        mlahead_channels=128,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')
