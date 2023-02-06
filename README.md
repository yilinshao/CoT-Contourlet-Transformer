# CoT: Contourlet Transformer for Hierarchical Semantic Segmentation
## Train
### 1. Train a CoT on ADE20K
```shell
python distributed/launch.py
--nproc_per_node=1
--master_port 29502
tools/train.py
configs/cot/cot_swin_v2_ct3_sparse_stemv2_lossv2_window12_freeze_ade20k_bs_16.py
--launcher
pytorch
```

### 2. Train a CoT on Cityscapes 
```shell
python distributed/launch.py
--nproc_per_node=1
--master_port 29502
tools/train.py
configs/cot/cot_swin_v2_ct3_sparse_stemv2_lossv2_window12_freeze_cityscapes_bs_8.py
--launcher
pytorch
```

### 3. Train a CoT on PASCAL Context
```shell
python distributed/launch.py
--nproc_per_node=1
--master_port 29502
tools/train.py
configs/cot/cot_swin_v2_ct3_sparse_stemv2_lossv2_weightsv2_window12_freeze_pascal_bs_16.py
--launcher
pytorch
```
## Evaluate
### Calculate evaluation mIoU
```shell
python distributed/launch.py
--nproc_per_node=4
--master_port
29504
tools/test.py
configs/cot/cot_swin_v2_ct3_sparse_stemv2_lossv2_window12_freeze_cityscapes_bs_8.py
work_dirs/cot_swin_v2_ct3_sparse_stemv2_lossv2_window12_freeze_cityscapes_bs_8/best.pth
--launcher
pytorch
--eval
mIoU
```

### Save predicted mask
```shell
python test.py 
configs/cot/cot_swin_v2_ct3_sparse_stemv2_lossv2_window12_freeze_ade20k_bs_16_ms.py
work_dirs/cot_swin_v2_ct3_sparse_stemv2_lossv2_window12_freeze_ade20k_bs_16/latest.pth
--eval
mIoU
--show-dir
predicted_mask/
```

