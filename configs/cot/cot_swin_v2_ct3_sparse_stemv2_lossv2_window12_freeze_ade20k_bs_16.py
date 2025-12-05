_base_ = [
    '../_base_/models/cot_swin_v2.py',
    '../_base_/datasets/ade20k_nsct.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
uper_swin_checkpoint_file = 'pretrain/converted_upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k.pth'
norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth'  # noqa

model = dict(
    freeze_swin=True,
    pretrained_uper_swin=uper_swin_checkpoint_file,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12),
    neck=dict(ct_levels=3,
              in_channels=[192, 384, 768, 1536],
              num_classes=150
              ),
    decode_head=dict(num_classes=150),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
    # test_cfg=dict(mode='whole'),

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
    ]
)

# optimizer = dict(lr=0.001, weight_decay=0.0,
#                  paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.),
#                                                  'resnet': dict(lr_mult=10.),
#                                                  'absolute_pos_embed': dict(decay_mult=0.),
#                                                  'relative_position_bias_table': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.)
#                                                  })
#                  )

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'head': dict(lr_mult=10.),
                     'resnet': dict(lr_mult=10.),
                     'absolute_pos_embed': dict(decay_mult=0.),
                     'relative_position_bias_table': dict(decay_mult=0.),
                     'norm': dict(decay_mult=0.)
                     }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

find_unused_parameters = True

data = dict(
    samples_per_gpu=4,
    train=dict(
        img_dir='images/training',
        ann_dir='annotations/training',
        nsct_dir='nsct_012/training'),
    val=dict(
        img_dir='images/validation',
        ann_dir='annotations/validation',
        nsct_dir='nsct_012/validation'),
    test=dict(
        img_dir='images/validation',
        ann_dir='annotations/validation',
        nsct_dir='nsct_012/validation')
)
