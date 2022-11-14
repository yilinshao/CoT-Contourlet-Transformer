_base_ = ['cot_swin_v2_sparse_window12_freeze_ade20k_bs_16.py']

model = dict(
    neck=dict(
        ct_levels=3,
    )
)

data = dict(
    samples_per_gpu=4,
    train=dict(
        nsct_dir='nsct/training'),
    val=dict(
        nsct_dir='nsct/validation'),
    test=dict(
        nsct_dir='nsct/validation')
)
