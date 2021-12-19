<<<<<<< HEAD
# CoT
=======
# CoT: Contourlet Transformer

## Environment

- Linux 18.04
- Python 3.8
- CUDA 11.4
- PyTorch 1.7.0
- mmcv-full 1.2.7
- opencv-python 4.5.3.56
- tqdm

## Prepare Dataset

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
We provided a [scripts](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/convert_datasets/cityscapes.py) based on [cityscapesscripts](https://github.com/mcordts/cityscapesScripts)
to generate `**labelTrainIds.png`.

```shell
# --nproc means 8 process for conversion, which could be omitted as well.
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```

### ADE20K

The training and validation set of ADE20K could be download from this [link](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip).
We may also download test set from [here](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip).

### Pascal Context

The training and validation set of Pascal Context could be download from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar). You may also download test set from [here](http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2010test.tar) after registration.

To split the training and validation set from original dataset, you may download trainval_merged.json from [here](https://codalabuser.blob.core.windows.net/public/trainval_merged.json).

If you would like to use Pascal Context dataset, please install [Detail](https://github.com/zhanghang1989/detail-api) and then run the following command to convert annotations into proper format.

```shell
python tools/convert_datasets/pascal_context.py data/VOCdevkit data/VOCdevkit/VOC2010/trainval_merged.json
```

## Contourlet Decomposition

Decompose the RGB image into multi-directional components:
```shell
cd tools/contourlet_transform
python generate_dec_img.py --dataset {DATASET_NAME} --levels {DECOMPOSITION_LEVELS}
```
For example, a decomposition of 3 level on Cityscapes dataset:
``` shell
python generate_dec_img.py --dataset cityscapes --levels 3
```

## Train 

```shell
python -m torch.distributed.launch --nproc_per_node=4 --master_port=-29500 tools/train.py 
{CONFIG_FILE_PATH} 
--launcher pytorch 
--gpus 4
```
The config files can be found under 
```shell
./configs/cot
```
And the state-of-the-art models are trained from the following config files
```shell
./configs/cotexp12_cot_ade20k_hfpe_01_bs_16_ms.py
./configs/cotexp13_cot_cityscapes_hfpe_bs_4_ms.py
./configs/cotexp14_cot_pascal_context_hfpe_bs_16_ms.py
``` 

## Inference
```shell
python -m torch.distributed.launch --nproc_per_node=4 --master_port=-29500 tools/test.py
{CONFIG_FILE_PATH}
{CHECKPOINT_FILE_PATH}
--eval
mIoU
--launcher
pytorch
```
>>>>>>> master
