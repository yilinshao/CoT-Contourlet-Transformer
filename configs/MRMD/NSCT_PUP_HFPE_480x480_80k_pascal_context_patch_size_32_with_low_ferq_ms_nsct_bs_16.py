_base_ = [
    '../_base_/models/nsct.py',
    '../_base_/datasets/pascal_context_nsct.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    backbone=dict(
        ms_nsct=True,
        vit_backbone=dict(hfpe=True, img_size=480, pos_embed_interp=True, drop_rate=0., patch_size=32,
                          use_low_freq=True, mla_channels=256, mla_index=(5, 11, 17, 23))),
    decode_head=dict(img_size=480, mla_channels=256,
                     mlahead_channels=128, num_classes=60),
    auxiliary_head=[
        dict(
            type='VIT_MLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=0,
            img_size=480,
            num_classes=60,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='VIT_MLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=1,
            img_size=480,
            num_classes=60,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='VIT_MLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=2,
            img_size=480,
            num_classes=60,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='VIT_MLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=3,
            img_size=480,
            num_classes=60,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    ])

optimizer = dict(lr=0.001, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})
                 )

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (520, 520)
crop_size = (480, 480)

test_cfg = dict(mode='slide', crop_size=crop_size, stride=(320, 320))
find_unused_parameters = True
data = dict(samples_per_gpu=4)

