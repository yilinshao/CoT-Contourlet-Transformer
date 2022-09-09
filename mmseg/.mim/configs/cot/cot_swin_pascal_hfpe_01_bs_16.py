_base_ = [
    '../_base_/models/cot_swin.py',
    '../_base_/datasets/pascal_context_nsct.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (480, 480)
model = dict(
    decode_head=dict(num_classes=60),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=0,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=60,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=1,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=60,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=2,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=60,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=3,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=60,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    ],
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(320, 320)),
)

optimizer = dict(lr=0.001, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})
                 )
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
find_unused_parameters = True
data = dict(
    samples_per_gpu=4,
    train=dict(
        nsct_dir='nsct_01'),
    val=dict(
        nsct_dir='nsct_01'),
    test=dict(
        nsct_dir='nsct_01')
)

