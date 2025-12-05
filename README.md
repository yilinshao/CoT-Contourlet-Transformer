# CoT: Contourlet Transformer for Hierarchical Semantic Segmentation

[](https://opensource.org/licenses/MIT)
[](https://www.google.com/search?q=https://ieeexplore.ieee.org/xpl/RecentIssue.jsp%3Fpunumber%3D5962385)

This repository contains the official implementation of the paper **"CoT: Contourlet Transformer for Hierarchical Semantic Segmentation"** (IEEE TNNLS).

Our method leverages the Contourlet Transform to achieve efficient hierarchical feature extraction for semantic segmentation tasks. 
This codebase is built upon [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [swin-transformer](https://github.com/microsoft/Swin-Transformer), and [spconv](https://github.com/traveller59/spconv).

-----

## 🛠️ Installation

Please ensure you meet the following requirements:

  * **OS**: Linux
  * **Python**: 3.8+
  * **PyTorch**: 1.11.0
  * **CUDA**: 11.3 / cuDNN 8

### 1\. Environment Setup

```bash
# Create a conda environment (recommended)
conda create -n cot python=3.8 -y
conda activate cot

# Install PyTorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install MIM and MMCV
pip install -U openmim
mim install mmcv-full==1.5.0

# Install spconv (Choose the version matching your CUDA)
pip install spconv-cu113
```

### 2\. Install CoT

```bash
git clone https://github.com/yilinshao/CoT-Contourlet-Transformer.git
cd CoT-Contourlet-Transformer
pip install -v -e .
```

-----

## 📂 Data Preparation

### 1\. Dataset Structure

Please organize your datasets in the `data/` directory as follows:

```none
CoT-Contourlet-Transformer
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

### 2\. Dataset Setup

  * **Cityscapes:** The data could be found [here](https://www.cityscapes-dataset.com/downloads/) after registration. 
By convention, `**labelTrainIds.png` are used for cityscapes training.
MMSegmentation provides a [scripts](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/convert_datasets/cityscapes.py) based on [cityscapesscripts](https://github.com/mcordts/cityscapesScripts)
to generate `**labelTrainIds.png`.
    ```bash
    python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
    ```
  * **ADE20K:** The training and validation set of ADE20K could be download from this [link](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip).
You may also download test set from [here](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip).

  * **Pascal Context:** The training and validation set of Pascal Context could be download from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar). 
You may also download test set from [here](http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2010test.tar) after registration. 
To split the training and validation set from original dataset, you may download trainval_merged.json from [here](https://codalabuser.blob.core.windows.net/public/trainval_merged.json). 
If you would like to use Pascal Context dataset, please install [Detail](https://github.com/zhanghang1989/detail-api) and then run the following command to convert annotations into proper format.
    ```shell
    python tools/convert_datasets/pascal_context.py data/VOCdevkit data/VOCdevkit/VOC2010/trainval_merged.json
    ```

### 3\. Contourlet Decomposition

The CoT requires pre-computed contourlet decomposition images.

1.  Modify `tools/contourlet_transform/tools/generate_dec_img.py` (and the `save_nsct()` function) to select your target dataset.
2.  Run the generation script:
    ```bash
    python tools/contourlet_transform/tools/generate_dec_img.py
    ```


-----

## 🚀 Usage

### 1\. Train Baseline (Swin Transformer)

First, train the vanilla Swin Transformer backbone or download the pre-trained weights.


**Example (ADE20K):**

```bash
# Using torch.distributed.launch directly
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py \
    configs/swin/upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k.py \
    --launcher pytorch
```
The configs implemented in the paper for Cityscapes and Pascal Context are 

### 2\. Convert Weights

Before training CoT, you must convert the Swin Transformer weights to the CoT format. Place your trained weights in the `./pretrain` folder.
Then run
```bash
python tools/convert_uper_swin_to_cot_swin.py
```

### 3\. Train CoT

Train the Contourlet Transformer using the converted weights.

**Command syntax:**

```bash
python -m torch.distributed.launch --nproc_per_node=<GPUS> tools/train.py <CONFIG_FILE> --launcher pytorch
```

**Configurations in the paper:**
 (note: `_ms` refers to the multi-scale version)

| Dataset | Config File Example                                                                               |
| :--- |:--------------------------------------------------------------------------------------------------|
| **ADE20K** | `configs/cot/cot_swin_v2_ct3_sparse_stemv2_lossv2_window12_freeze_ade20k_bs_16(_ms).py`           |
| **Cityscapes** | `configs/cot/cot_swin_v2_ct3_sparse_stemv2_lossv2_window12_freeze_cityscapes_bs_8(_ms).py`        |
| **Pascal Context** | `configs/cot/cot_swin_v2_ct3_sparse_stemv2_lossv2_weightsv2_window12_freeze_pascal_bs_16(_ms).py` |

### 4\. Evaluation & Inference

**Calculate mIoU:**

```bash
python -m torch.distributed.launch --nproc_per_node=2 tools/test.py \
    <CONFIG_FILE> \
    <CHECKPOINT_FILE> \
    --launcher pytorch \
    --eval mIoU
```

**Visualize/Save Masks:**

```bash
python tools/test.py \
    <CONFIG_FILE> \
    <CHECKPOINT_FILE> \
    --eval mIoU \
    --show-dir predicted_mask/
```

-----

## 🤝 Acknowledgements

This project is primarily based on [MMSegmentation v0.26](https://github.com/open-mmlab/mmsegmentation). We gratefully acknowledge the authors of the following open-source projects:

  * **MMSegmentation**: OpenMMLab Semantic Segmentation Toolbox and Benchmark.
  * **Swin Transformer**: Hierarchical Vision Transformer using Shifted Windows.
  * **spconv**: Spatially Sparse Convolution Library.

-----

## ✏️ Citation

If you find this work helpful in your research, please consider citing our TNNLS paper:

```bibtex
@article{CoT_TNNLS2024,
  title={CoT: Contourlet Transformer for Hierarchical Semantic Segmentation},
  author={Your Name and Co-authors},
  journal={IEEE Transactions on Neural Networks and Learning Systems (TNNLS)},
  year={2024},
  publisher={IEEE}
}
```