# GeminiFusion: Efficient Pixel-wise Multimodal Fusion for Vision Transformer 

<!-- **News**:
 * :fire: June, 2023. TR3D is accepted at [ICIP2023](https://2023.ieeeicip.org/).
 * :rocket: June, 2023. We add ScanNet-pretrained S3DIS model and log significantly pushing forward state-of-the-art.
 * February, 2023. TR3D on all 3 datasets is now supported in [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) as a [project](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/TR3D).
 * :fire: February, 2023. TR3D is now state-of-the-art on [paperswithcode](https://paperswithcode.com) on SUN RGB-D and S3DIS. -->

This repository contains an implementation of GeminiFusion on the MVX-Net codebase, a 3D object detection method introduced in our paper:

> **GeminiFusion: Efficient Pixel-wise Multimodal Fusion for Vision Transformer**<br>
<!-- > [Danila Rukhovich](https://github.com/filaPro),
> [Anna Vorontsova](https://github.com/highrut),
> [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ)
> <br>
> Samsung Research<br>
> https://arxiv.org/abs/2302.02858 -->

## Installation
<!-- For convenience, we provide a [Dockerfile](docker/Dockerfile).

Alternatively, you can install all required packages manually. This implementation is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) framework.
Please refer to the original installation guide [getting_started.md](docs/en/getting_started.md), including MinkowskiEngine installation, replacing `open-mmlab/mmdetection3d` with `samsunglabs/tr3d`. -->
We build our geminifusion on the MMDet3d codebase. Therefore, the installation is exactly the same as the MMDet3d. Please refer to [the offical install instructions](https://mmdetection3d.readthedocs.io/en/latest/get_started.html). Also, you may refer to [the offical MMDet3D readme](./README_mmdet3d.md) if any problem about the framework.

Most of the `GeminiFusion`-related code locates in the following files: 
[fusion_layers/point_fusion.py](mmdet3d/models/layers/fusion_layers/point_fusion.py),
[voxel_encoders/voxel_encoder.py](mmdet3d/models/voxel_encoders/voxel_encoder.py).

## Getting Started

**KITTI Dataset Prapare**

Please follow [the mmdetection3d data preparation instructions for kitti](https://mmdetection3d.readthedocs.io/en/v1.1.0/user_guides/dataset_prepare.html#kitti).

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

Test after the training finished.

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

We used the results of the last epoch for all experiments.(The numbers are from the ckpt, **Rerunning now! Maybe use the new ckpt and logs and results!**)                                             

### 3D AP R11(IoU=0.7) 

| Model |Param| Easy | Moderate | Hard | Download |
|:-------:|:--------:|:--------:|:-------:|:-------------------:|:--------:|
| MVX-Net(Offical) | 33.8M|87.49| 77.04| 74.54 | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-20230424_132228.log) &#124; [config](configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py) |
| MVX-Net + GeminiFusion |34.8M|  88.49(+1.0)| 77.36(+0.32)| 74.61(+0.07) | [model](https://github.com/JiaDingCN/GeminiFusion/releases/download/CheckPoint_and_Log/geminifusion_mvxnet.pth) &#124; [log](https://github.com/JiaDingCN/GeminiFusion/releases/download/CheckPoint_and_Log/geminifusion_mvxnet_training.log) &#124; [config](configs/geminifusion_mvxnet/geminifusion_mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py) |

### 3D AP R40(IoU=0.7) 

| Model | Param|Easy | Moderate | Hard | Download |
|:-------:|:--------:|:--------:|:-------:|:-------------------:|:--------:|
| MVX-Net(Offical) |33.8M| 88.41| 78.77| 74.27  | [model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-20230424_132228.log) &#124; [config](configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py) |
| MVX-Net + GeminiFusion | 34.8M| 89.43(+1.02)| 78.76(+0.01)| 74.46(+0.19)  | [model](https://github.com/JiaDingCN/GeminiFusion/releases/download/CheckPoint_and_Log/geminifusion_mvxnet.pth) &#124; [log](https://github.com/JiaDingCN/GeminiFusion/releases/download/CheckPoint_and_Log/geminifusion_mvxnet_training.log) &#124; [config](configs/geminifusion_mvxnet/geminifusion_mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py) |

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
