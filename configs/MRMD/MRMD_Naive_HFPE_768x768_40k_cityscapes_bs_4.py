_base_ = [
    '../_base_/models/mrmd.py',
    '../_base_/datasets/cityscapes_768x768.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        vit_backbone=dict(
            high_freq_pos_embed=True)))

norm_cfg = dict(type='SyncBN', requires_grad=True)
optimizer = dict(lr=0.01, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})
                 )
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (768, 768)
test_cfg = dict(mode='slide', crop_size=crop_size, stride=(512, 512))
find_unused_parameters = True
data = dict(samples_per_gpu=1)


