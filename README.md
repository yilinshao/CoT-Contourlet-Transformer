# CoT: Contourlet Transformer for Hierarchical Semantic Segmentation

## Step-0安装
1. torch环境: torch:1.11.0 cuda11.3 cudnn8
2. 安装openmim 和mmcv
(安装pip install -U openmim mim install mmcv-full==1.5.0)
3. clone 本项目(基于mmsegmentation v0.26)
4. 安装 mmsegmentation pip install

## Step-1数据集准备
./data文件夹, 数据集结构如下

The datasets should be arranged to the following structure:
```none
CoT
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2010
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClassContext
│   │   │   ├── ImageSets
│   │   │   │   ├── SegmentationContext
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── val.txt
│   │   │   ├── trainval_merged.json
│   │   ├── VOCaug
│   │   │   ├── dataset
│   │   │   │   ├── cls
│   ├── ade
│   │   ├── ADEChallengeData2016
│   │   │   ├── annotations
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   │   │   ├── images
│   │   │   │   ├── training
│   │   │   │   ├── validation
```

### Cityscapes

The data could be found [here](https://www.cityscapes-dataset.com/downloads/) after registration.

By convention, `**labelTrainIds.png` are used for cityscapes training.
MMSegmentation provides a [scripts](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/convert_datasets/cityscapes.py) based on [cityscapesscripts](https://github.com/mcordts/cityscapesScripts)
to generate `**labelTrainIds.png`.

```shell
# --nproc means 8 process for conversion, which could be omitted as well.
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```

### ADE20K

The training and validation set of ADE20K could be download from this [link](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip).
You may also download test set from [here](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip).

### Pascal Context

The training and validation set of Pascal Context could be download from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar). You may also download test set from [here](http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2010test.tar) after registration.

To split the training and validation set from original dataset, you may download trainval_merged.json from [here](https://codalabuser.blob.core.windows.net/public/trainval_merged.json).

If you would like to use Pascal Context dataset, please install [Detail](https://github.com/zhanghang1989/detail-api) and then run the following command to convert annotations into proper format.

```shell
python tools/convert_datasets/pascal_context.py data/VOCdevkit data/VOCdevkit/VOC2010/trainval_merged.json
```

## Step2- Preprocess: Contourlet Decompose
```shell
python tools/contourlet_transform/tools/generate_dec_img.py
```
修改generate_dec_img.py和save_nsct()函数,来对不同数据集进行分解并保存

## Step3 - Train Contourlet Transformer
### Step3.1 - Train the vanilla swin-transformer
```shell
/opt/conda/bin/python -m torch.distributed.launch \
--nproc_per_node=2 \
tools/train.py \
configs/swin/{upnet_swin_config_name.py} \
--launcher pytorch
```
--nproc_per_node=2表示GPU数量(要求>=2). You can replace the {upnet_swin_config_name.py} with ```upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k.py```
or ```converted_upernet_swin_large_patch4_window12_512x1024_pretrain_384x384_22K_80k_cityscapes.py``` or ```converted_upernet_swin_large_patch4_window12_480x480_pretrain_384x384_22K_80k_pascal_context.py```
for training on different datasets, 将训练权重放入./pretrain文件夹, 然后通过```convert_uper_swin_to_cot_swin.py``将权重进行转换

### Step 3.2 - Train CoT
```shell
python -m torch.distributed.launch \
--nproc_per_node=2 \
tools/train.py \
configs/cot/{cot_config_name.py} --launcher pytorch
```
You can replace the {cot_config_name.py} with ```cot_swin_v2_ct3_sparse_stemv2_lossv2_window12_freeze_ade20k_bs_16.py```
or ```cot_swin_v2_ct3_sparse_stemv2_lossv2_window12_freeze_cityscapes_bs_8.py```
or ```cot_swin_v2_ct3_sparse_stemv2_lossv2_weightsv2_window12_freeze_pascal_bs_16.py```


## Step 3.3 Evaluate & Test
### Calculate evaluation mIoU
```shell
python -m torch.distributed.launch \
--nproc_per_node=2 \
tools/test.py \
configs/cot/{cot_config_name.py} \
work_dirs/{cot_config_name}/best.pth \
--launcher pytorch \
--eval mIoU
```

### Save predicted mask
```shell
python test.py \
configs/cot/{cot_config_name.py} \
work_dirs/{cot_config_name}/latest.pth \
--eval mIoU \
--show-dir predicted_mask/
```

