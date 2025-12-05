# CoT: Contourlet Transformer for Hierarchical Semantic Segmentation

## 安装
1. 选择镜像: torch:1.11.0-cuda11.3-cudnn8
2. 下载mmsegmentation v0.26
3. 根据的get_started安装openmim 和mmcv
(安装pip install -U openmim mim install mmcv-full==1.5.0)
4. 安装 mmsegmentation
5. 将 CoT 代码复制进去

## 数据集准备
./data文件夹, 数据集结构如下



## Pre-train uper-swin
将训练权重放入pretrain文件夹
## Contourlet decompose
## Train CoT
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

