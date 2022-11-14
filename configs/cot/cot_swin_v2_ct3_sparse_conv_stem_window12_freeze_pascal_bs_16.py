_base_ = ['cot_swin_v2_ct3_sparse_window12_freeze_pascal_bs_16.py']

model = dict(
    neck = dict(
        sparse_resnet_config=dict(
            use_conv_in_stem=True
        )
    )
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                     'relative_position_bias_table': dict(decay_mult=0.),
                     'norm': dict(decay_mult=0.)
                     }))
