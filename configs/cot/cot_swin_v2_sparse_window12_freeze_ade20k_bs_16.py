_base_ = ['cot_swin_v2_sparse_ade20k_bs_16.py']
uper_swin_checkpoint_file = 'pretrain/converted_upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k.pth'
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'  # noqa
norm_cfg = dict(type='SyncBN', requires_grad=True)


model = dict(
    freeze_swin=True,
    pretrained_uper_swin=uper_swin_checkpoint_file,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        window_size=12),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=512,
            in_index=1,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.3)
        ),
        dict(
            type='FCNHead',
            in_channels=512,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.3)
        ),
        dict(
            type='FCNHead',
            in_channels=512,
            in_index=3,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.3)
        ),
        dict(
            type='PseudoHead',
            in_channels=150,
            in_index=4,
            channels=150,
            num_convs=0,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.0)
        ),
    ],
)
