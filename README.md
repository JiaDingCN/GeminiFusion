# [ICML2024] GeminiFusion <br> for Multimodal 3D Object Detection on KITTI Dataset

This is the official implementation of our paper "[GeminiFusion: Efficient Pixel-wise Multimodal Fusion for Vision Transformer](Link)".

Authors: Ding Jia, Jianyuan Guo, Kai Han, Han Wu, Chao Zhang, Chang Xu, Xinghao Chen

----------------------------

## Code List

We have applied our GeminiFusion to different tasks and datasets:

* GeminiFusion for Multimodal Semantic Segmentation
  * [NYUDv2 & SUN RGBD datasets](https://github.com/JiaDingCN/GeminiFusion/tree/main)
  * [DeLiVER dataset](https://github.com/JiaDingCN/GeminiFusion/tree/DeLiVER)
* GeminiFusion for Multimodal 3D Object Detection
  * (This branch)[KITTI dataset](https://github.com/JiaDingCN/GeminiFusion/tree/3d_object_detection_kitti)
----------------

## Installation

We build our GeminiFusion on the MMDetection3D codebase. Therefore, the installation is exactly the same as the MMDetection3D. Please refer to [the offical MMDetection3D install instructions](https://mmdetection3d.readthedocs.io/en/latest/get_started.html). Also, you may refer to [the offical MMDetection3D readme](./README_mmdet3d.md) if any problem about the MMDetection3D framework.

Most of the `GeminiFusion`-related code locate in the following files: 
* [config/geminifusion_mvxnet](configs/geminifusion_mvxnet/geminifusion_mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py)
* [fusion_layers/point_fusion.py](mmdet3d/models/layers/fusion_layers/point_fusion.py)
* [voxel_encoders/voxel_encoder.py](mmdet3d/models/voxel_encoders/voxel_encoder.py)

## Getting Started

**KITTI Dataset Prapare**

Please follow [the MMDetection3D data preparation instructions for KITTI](https://mmdetection3d.readthedocs.io/en/v1.1.0/user_guides/dataset_prepare.html#kitti).

**Training**

We use 4 GPUs to train the GeminiFusion_MVX-Net.
```shell
bash ./tools/dist_train.sh configs/geminifusion_mvxnet/geminifusion_mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py 4
```

**Testing**

Test the downloaded checkpoint. The download link is below.
```shell
bash ./tools/dist_test.sh configs/geminifusion_mvxnet/geminifusion_mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py geminifusion_mvxnet.pth 4 
```

Test after your training finished.
```shell
bash ./tools/dist_test.sh configs/geminifusion_mvxnet/geminifusion_mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py work_dirs/geminifusion_mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class/epoch_40.pth 4 
```

You can also get the number of params by the command below:
```Shell
# param, geminifusion_mvxnet
python ./tools/analysis_tools/get_flops.py configs/geminifusion_mvxnet/geminifusion_mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py

# param, mvxnet
python ./tools/analysis_tools/get_flops.py configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py
```

## Models on the KITTI validation set for vehicle detection, under the evaluation metric of 3D Average Precision (AP)

We used the results of the last epoch for all experiments.(The numbers are from the ckpt)                                             

### 3D AP R11(IoU=0.7) 

| Model |Param| Easy | Moderate | Hard | Download |
|:-------:|:--------:|:--------:|:-------:|:-------------------:|:--------:|
| MVX-Net(Offical) | 33.8M|87.49| 77.04| 74.54 | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-20230424_132228.log) &#124; [config](configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py) |
| MVX-Net + GeminiFusion |34.8M|  88.49| 77.36| 74.61 | [model](https://github.com/JiaDingCN/GeminiFusion/releases/download/CheckPoint_and_Log/geminifusion_mvxnet.pth) &#124; [log](https://github.com/JiaDingCN/GeminiFusion/releases/download/CheckPoint_and_Log/geminifusion_mvxnet_training.log) &#124; [config](configs/geminifusion_mvxnet/geminifusion_mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py) |

### 3D AP R40(IoU=0.7) 

| Model | Param|Easy | Moderate | Hard | Download |
|:-------:|:--------:|:--------:|:-------:|:-------------------:|:--------:|
| MVX-Net(Offical) |33.8M| 88.41| 78.77| 74.27  | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-20230424_132228.log) &#124; [config](configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py) |
| MVX-Net + GeminiFusion | 34.8M| 89.43| 78.76| 74.46  | [model](https://github.com/JiaDingCN/GeminiFusion/releases/download/CheckPoint_and_Log/geminifusion_mvxnet.pth) &#124; [log](https://github.com/JiaDingCN/GeminiFusion/releases/download/CheckPoint_and_Log/geminifusion_mvxnet_training.log) &#124; [config](configs/geminifusion_mvxnet/geminifusion_mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py) |

### Citation

If you find this work useful for your research, please cite our paper:

<!-- ```
@misc{rukhovich2023tr3d,
  doi = {10.48550/ARXIV.2302.02858},
  url = {https://arxiv.org/abs/2302.02858},
  author = {Rukhovich, Danila and Vorontsova, Anna and Konushin, Anton},
  title = {TR3D: Towards Real-Time Indoor 3D Object Detection},
  publisher = {arXiv},
  year = {2023}
}
``` -->
