_base_ = [
    '../_base_/models/psanet_r50-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadNsctNpy'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    val=dict(type='CityscapesDatasetWithNSCT',
             img_dir='leftImg8bit_fog_severity_3/val',
             nsct_dir='nsct/val',
             nsct_suffix='_leftImg8bit.npy',
             pipeline=test_pipeline),
    test=dict(type='CityscapesDatasetWithNSCT',
             img_dir='leftImg8bit_fog_severity_3/val',
             nsct_dir='nsct/val',
             nsct_suffix='_leftImg8bit.npy',
             pipeline=test_pipeline))