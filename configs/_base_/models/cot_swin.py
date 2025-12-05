backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth'  # noqa

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ContourletTransformer',
        norm_cfg=norm_cfg,
        vit_backbone=dict(
            type='SwinTransformer',
            pretrain_img_size=224,
            embed_dims=192,
            patch_size=4,
            window_size=7,
            mlp_ratio=4,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            strides=(4, 2, 2, 2),
            out_indices=(0, 1, 2, 3),
            qkv_bias=True,
            qk_scale=None,
            patch_norm=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            use_abs_pos_embed=False,
            act_cfg=dict(type='GELU'),
            norm_cfg=backbone_norm_cfg,
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        ),
        resnet_backbone_1=dict(
            type='ResNet',
            depth=50,
            in_channels=1,
            num_stages=2,
            out_indices=(1, ),
            dilations=(1, 1),
            strides=(1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            init_cfg=dict(type='Pretrained', checkpoint='pretrain/resnet50.pth')
        ),
        resnet_backbone_2=dict(
            type='ResNet',
            depth=50,
            in_channels=2,
            num_stages=2,
            out_indices=(1, ),
            dilations=(1, 1),
            strides=(1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            init_cfg=dict(type='Pretrained', checkpoint='pretrain/resnet50.pth')
        )
    ),
    neck=dict(
        type='CoTNeck',
        setr_mla_neck=dict(
            type='MLANeck',
            in_channels=[192, 384, 768, 1536],
            out_channels=256,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
        ),
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
    ),
    decode_head=dict(
            type='SETRMLAHead',
            in_channels=(256, 256, 256),
            channels=384,
            in_index=(1, 2, 3),
            dropout_ratio=0,
            mla_channels=128,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
            dict(
                type='FCNHead',
                in_channels=256,
                channels=256,
                in_index=0,
                dropout_ratio=0,
                num_convs=0,
                kernel_size=1,
                concat_input=False,
                num_classes=19,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
            dict(
                type='FCNHead',
                in_channels=256,
                channels=256,
                in_index=1,
                dropout_ratio=0,
                num_convs=0,
                kernel_size=1,
                concat_input=False,
                num_classes=19,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
            dict(
                type='FCNHead',
                in_channels=256,
                channels=256,
                in_index=2,
                dropout_ratio=0,
                num_convs=0,
                kernel_size=1,
                concat_input=False,
                num_classes=19,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
            dict(
                type='FCNHead',
                in_channels=256,
                channels=256,
                in_index=3,
                dropout_ratio=0,
                num_convs=0,
                kernel_size=1,
                concat_input=False,
                num_classes=19,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

