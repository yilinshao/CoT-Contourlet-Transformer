_base_ = [
    '../_base_/models/cot_swin_v2.py',
    '../_base_/datasets/ade20k_nsct.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth'  # noqa

model = dict(
    freeze_swin=True,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=224,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7),
    neck=dict(in_channels=[192, 384, 768, 1536], num_classes=150),
    decode_head=dict(num_classes=150),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
)

optimizer = dict(lr=0.001, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.),
                                                 'resnet': dict(lr_mult=10.),
                                                 'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)
                                                 })
                 )

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
find_unused_parameters = True

data = dict(
    samples_per_gpu=4,
    train=dict(
        nsct_dir='nsct_01/training'),
    val=dict(
        nsct_dir='nsct_01/validation'),
    test=dict(
        nsct_dir='nsct_01/validation')
)

