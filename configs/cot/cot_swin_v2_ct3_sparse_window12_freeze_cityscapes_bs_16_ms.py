_base_ = ['cot_swin_v2_ct3_sparse_window12_freeze_cityscapes_bs_16.py']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadNsctNpy'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='ResizeWithNSCT', keep_ratio=True),
            dict(type='RandomFlipWithNSCT'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img', 'nsct_feature']),
            dict(type='Collect', keys=['img', 'nsct_feature']),
        ])
]

data = dict(
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))