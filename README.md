<div align="center"> 

##  GeminiFusion for Multimodal Segementation on NYUDv2 & SUN RGBD Dataset (ICML 2024)

</div>

<p align="center">

<a href="https://arxiv.org/pdf/2406.01210">
    <img src="https://img.shields.io/badge/arXiv-2406.01210-green" /></a>
<a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange.svg" /></a>
<a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" /></a>

</p>

	
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/geminifusion-efficient-pixel-wise-multimodal/semantic-segmentation-on-deliver-1)](https://paperswithcode.com/sota/semantic-segmentation-on-deliver-1?p=geminifusion-efficient-pixel-wise-multimodal)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/geminifusion-efficient-pixel-wise-multimodal/semantic-segmentation-on-nyu-depth-v2)](https://paperswithcode.com/sota/semantic-segmentation-on-nyu-depth-v2?p=geminifusion-efficient-pixel-wise-multimodal)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/geminifusion-efficient-pixel-wise-multimodal/semantic-segmentation-on-sun-rgbd)](https://paperswithcode.com/sota/semantic-segmentation-on-sun-rgbd?p=geminifusion-efficient-pixel-wise-multimodal)


This is the official implementation of our paper "[GeminiFusion: Efficient Pixel-wise Multimodal Fusion for Vision Transformer](https://arxiv.org/pdf/2406.01210)".

Authors: Ding Jia, Jianyuan Guo, Kai Han, Han Wu, Chao Zhang, Chang Xu, Xinghao Chen



## Code List

We have applied our GeminiFusion to different tasks and datasets:

* GeminiFusion for Multimodal Semantic Segmentation
  * (This branch)[NYUDv2 & SUN RGBD datasets](https://github.com/JiaDingCN/GeminiFusion/tree/main)
  * [DeLiVER dataset](https://github.com/JiaDingCN/GeminiFusion/tree/DeLiVER)
* GeminiFusion for Multimodal 3D Object Detection
  * [KITTI dataset](https://github.com/JiaDingCN/GeminiFusion/tree/3d_object_detection_kitti)


## Introduction

We propose GeminiFusion, a pixel-wise fusion approach that capitalizes on aligned cross-modal representations. GeminiFusion elegantly combines intra-modal and inter-modal attentions, dynamically integrating complementary information across modalities. We employ a layer-adaptive noise to adaptively control their interplay on a per-layer basis, thereby achieving a harmonized fusion process. Notably, GeminiFusion maintains linear complexity with respect to the number of input tokens, ensuring this multimodal framework operates with efficiency comparable to unimodal networks. Comprehensive evaluations demonstrate the superior performance of our GeminiFusion against leading-edge techniques.



## Framework
![geminifusion_framework](figs/geminifusion_framework.png)



## Model Zoo                                           

### NYUDv2 dataset

| Model | backbone| mIoU | Download |
|:-------:|:--------:|:-------:|:-------------------:|
| GeminiFusion | MiT-B3| 56.8 |  [model](https://github.com/JiaDingCN/GeminiFusion/releases/download/NYUDv2_V2/mit-b3.pth.tar)  |
| GeminiFusion | MiT-B5| 57.7 |  [model](https://github.com/JiaDingCN/GeminiFusion/releases/download/NYUDv2_V2/mit_b5.pth.tar)  |
| GeminiFusion | swin_tiny| 52.2 |  [model](https://github.com/JiaDingCN/GeminiFusion/releases/download/NYUDv2_V2/swin_tiny.pth.tar)  |
| GeminiFusion | swin-small| 55.0 |  [model](https://github.com/JiaDingCN/GeminiFusion/releases/download/NYUDv2_V2/swin_small.pth.tar)  |
| GeminiFusion | swin-large-224| 58.8 |  [model](https://github.com/JiaDingCN/GeminiFusion/releases/download/NYUDv2_V2/swin_large.pth.tar)  |
| GeminiFusion | swin-large-384| 60.2 |  [model](https://github.com/JiaDingCN/GeminiFusion/releases/download/NYUDv2_V2/swin_large_384.pth.tar)  |
| GeminiFusion | swin-large-384 +FineTune from SUN 300eps| 60.9 |  [model](https://github.com/JiaDingCN/GeminiFusion/releases/download/NYUDv2_V2/finetune-swin-large-384.pth.tar)  |

### SUN RGBD dataset

| Model | backbone| mIoU | Download |
|:-------:|:--------:|:-------:|:-------------------:|
| GeminiFusion | MiT-B3| 52.7 |  [model](https://github.com/JiaDingCN/GeminiFusion/releases/download/SUN_v2/mit-b3.pth.tar)  |
| GeminiFusion | MiT-B5| 53.3 |  [model](https://github.com/JiaDingCN/GeminiFusion/releases/download/SUN_v2/mit_b5.pth.tar)  |
| GeminiFusion | swin_tiny| 50.2 |  [model](https://github.com/JiaDingCN/GeminiFusion/releases/download/SUN_v2/swin_tiny.pth.tar)  |
| GeminiFusion | swin-large-384| 54.8 |  [model](https://github.com/JiaDingCN/GeminiFusion/releases/download/SUN_v2/swin-large-384.pth.tar)  |



## Installation

We build our GeminiFusion on the TokenFusion codebase, which requires no additional installation steps. If any problem about the framework, you may refer to [the offical TokenFusion readme](./README-TokenFusion.md).

Most of the `GeminiFusion`-related code locate in the following files: 
* [models/mix_transformer](models/mix_transformer.py): implement the GeminiFusion module for MiT backbones.
* [models/swin_transformer](models/swin_transformer.py):implement the GeminiFusion module for Swin backbones.
* [mmcv_custom](mmcv_custom): load checkpoints for Swin backbones.
* [main](main.py): enable SUN RGBD dataset.
* [utils/datasets](utils/datasets.py): enable SUN RGBD dataset.

We also delete the config.py in the TokenFusion codebase since it is not used here.



## Data

**NYUDv2 Dataset Prapare**

Please follow [the data preparation instructions for NYUDv2 in TokenFusion readme](./README-TokenFusion.md#datasets). In default the data path is `/cache/datasets/nyudv2`, you may change it by `--train-dir <your data path>`.

**SUN RGBD Dataset Prapare**

Please download the SUN RGBD dataset follow the link in [DFormer](https://github.com/VCIP-RGBD/DFormer?tab=readme-ov-file#2--get-start).In default the data path is `/cache/datasets/sunrgbd_Dformer/SUNRGBD`, you may change it by `--train-dir <your data path>`.



## Train

**NYUDv2 Training**

On the NYUDv2 dataset, we follow the TokenFusion's setting, using 3 GPUs to train the GeminiFusion. 

```shell
# mit-b3
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3  --use_env main.py --backbone mit_b3 --dataset nyudv2 -c nyudv2_mit_b3 

# mit-b5
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3  --use_env main.py --backbone mit_b5 --dataset nyudv2 -c nyudv2_mit_b5 --dpr 0.35

# swin_tiny
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3  --use_env main.py --backbone swin_tiny --dataset nyudv2 -c nyudv2_swin_tiny --dpr 0.2

# swin_small
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3  --use_env main.py --backbone swin_small --dataset nyudv2 -c nyudv2_swin_small

# swin_large
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3  --use_env main.py --backbone swin_large --dataset nyudv2 -c nyudv2_swin_large

# swin_large_window12
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3  --use_env main.py --backbone swin_large_window12 --dataset nyudv2 -c nyudv2_swin_large_window12 --dpr 0.2

# swin-large-384+FineTune from SUN 300eps
# swin-large-384.pth.tar should be downloaded by our link or trained by yourself
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3  --use_env main.py --backbone swin_large_window12 --dataset nyudv2 -c swin_large_window12_finetune_dpr0.15_100+200+100 \
 --dpr 0.15 --num-epoch 100 200 100 --is_pretrain_finetune --resume ./swin-large-384.pth.tar
```

**SUN RGBD Training**

On the SUN RGBD dataset, we use 4 GPUs to train the GeminiFusion. 
```shell
# mit-b3
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --use_env main.py --backbone mit_b3 --dataset sunrgbd --train-dir /cache/datasets/sunrgbd_Dformer/SUNRGBD -c sunrgbd_mit_b3

# mit-b5
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --use_env main.py --backbone mit_b5 --dataset sunrgbd --train-dir /cache/datasets/sunrgbd_Dformer/SUNRGBD -c sunrgbd_mit_b5 --weight_decay 0.05

# swin_tiny
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --use_env main.py --backbone swin_tiny --dataset sunrgbd --train-dir /cache/datasets/sunrgbd_Dformer/SUNRGBD -c sunrgbd_swin_tiny

# swin_large_window12
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --use_env main.py --backbone swin_large_window12 --dataset sunrgbd --train-dir /cache/datasets/sunrgbd_Dformer/SUNRGBD -c sunrgbd_swin_large_window12
```



## Test

To evaluate checkpoints, you need to add `--eval --resume <checkpoint path>` after the training script. 

For example, on the NYUDv2 dataset, the training script for GeminiFusion with mit-b3 backbone is:
```shell
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3  --use_env main.py --backbone mit_b3 --dataset nyudv2 -c nyudv2_mit_b3
```

To evaluate the trained or downloaded checkpoint, the eval script is:
```shell
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3  --use_env main.py --backbone mit_b3 --dataset nyudv2 -c nyudv2_mit_b3 --eval --resume mit-b3.pth.tar
```



## Citation

If you find this work useful for your research, please cite our paper:

```
@misc{jia2024geminifusion,
      title={GeminiFusion: Efficient Pixel-wise Multimodal Fusion for Vision Transformer}, 
      author={Ding Jia and Jianyuan Guo and Kai Han and Han Wu and Chao Zhang and Chang Xu and Xinghao Chen},
      year={2024},
      eprint={2406.01210},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



## Acknowledgement
Part of our code is based on the open-source project [TokenFusion](https://github.com/yikaiw/TokenFusion).
