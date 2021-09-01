_base_ = './cityscapes.py'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (768, 768)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadNsctNpy'),

    dict(type='ResizeWithNSCT', img_scale=(2049, 1025), ratio_range=(0.5, 2.0)),
    dict(type='RandomCropWithNSCT', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlipWithNSCT', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PadWithNSCT', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadNsctNpy'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2049, 1025),
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
dataset_type = 'CityscapesDatasetWithNSCT'
data = dict(
    train=dict(type=dataset_type,
               nsct_dir='nsct',
               nsct_suffix='.npy',
               pipeline=train_pipeline),
    val=dict(type=dataset_type,
             nsct_dir='nsct',
             nsct_suffix='.npy',
             pipeline=test_pipeline),
    test=dict(type=dataset_type,
              nsct_dir='nsct',
              nsct_suffix='.npy',
              pipeline=test_pipeline))
