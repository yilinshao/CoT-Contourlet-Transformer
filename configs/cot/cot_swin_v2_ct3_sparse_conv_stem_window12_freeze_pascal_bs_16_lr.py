_base_ = ['cot_swin_v2_ct3_sparse_window12_freeze_pascal_bs_16.py']

model = dict(
    neck = dict(
        sparse_resnet_config=dict(
            use_conv_in_stem=True
        )
    )
)