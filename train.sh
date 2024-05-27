# # NYU
# # b3
# CUDA_VISIBLE_DEVICES=4,5,6 python -m torch.distributed.launch --nproc_per_node=3 --master_port 8848 --use_env main.py --backbone mit_b3 --dataset nyudv2 -c nyudv2_mit_b3 

# # b5
# CUDA_VISIBLE_DEVICES=4,5,6 python -m torch.distributed.launch --nproc_per_node=3 --master_port 8848 --use_env main.py --backbone mit_b5 --dataset nyudv2 -c nyudv2_mit_b5 --dpr 0.35

# # swin_tiny
# CUDA_VISIBLE_DEVICES=4,5,6 python -m torch.distributed.launch --nproc_per_node=3 --master_port 8848 --use_env main.py --backbone swin_tiny --dataset nyudv2 -c nyudv2_swin_tiny --dpr 0.2

# # swin_small
# CUDA_VISIBLE_DEVICES=4,5,6 python -m torch.distributed.launch --nproc_per_node=3 --master_port 8848 --use_env main.py --backbone swin_small --dataset nyudv2 -c nyudv2_swin_small

# # swin_large
# CUDA_VISIBLE_DEVICES=4,5,6 python -m torch.distributed.launch --nproc_per_node=3 --master_port 8848 --use_env main.py --backbone swin_large --dataset nyudv2 -c nyudv2_swin_large

# # swin_large_window12
# CUDA_VISIBLE_DEVICES=4,5,6 python -m torch.distributed.launch --nproc_per_node=3 --master_port 8848 --use_env main.py --backbone swin_large_window12 --dataset nyudv2 -c nyudv2_swin_large_window12 --dpr 0.2

# # swin-large-384+dpr0.15+100-200-100+FineTune from SUN 300eps, 54.8
# CUDA_VISIBLE_DEVICES=4,5,6 python -m torch.distributed.launch --nproc_per_node=3 --master_port 8848 --use_env main.py --backbone swin_large_window12 --dataset nyudv2 -c rerun_54.8_swin_large_window12_finetune_dpr0.15_100+200+100 \
#  --dpr 0.15 --num-epoch 100 200 100 --is_pretrain_finetune --resume ./sun_54.8.pth.tar

# # swin-large-384+dpr0.15+100-200-100+FineTune from SUN 300eps, 54.6
# CUDA_VISIBLE_DEVICES=4,5,6 python -m torch.distributed.launch --nproc_per_node=3 --master_port 8848 --use_env main.py --backbone swin_large_window12 --dataset nyudv2 -c rerun_54.6_swin_large_window12_finetune_dpr0.15_100+200+100 \
#  --dpr 0.15 --num-epoch 100 200 100 --is_pretrain_finetune --resume ./sun_54.6.pth.tar



# # SUN
# # b3
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8848 --use_env main.py --backbone mit_b3 --dataset sunrgbd --train-dir /cache/datasets/sunrgbd_Dformer/SUNRGBD -c sunrgbd_mit_b3

# # b5
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8848 --use_env main.py --backbone mit_b5 --dataset sunrgbd --train-dir /cache/datasets/sunrgbd_Dformer/SUNRGBD -c sunrgbd_mit_b5 --weight_decay 0.05

# # swin_tiny
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8848 --use_env main.py --backbone swin_tiny --dataset sunrgbd --train-dir /cache/datasets/sunrgbd_Dformer/SUNRGBD -c sunrgbd_swin_tiny

# # swin_large_window12
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8848 --use_env main.py --backbone swin_large_window12 --dataset sunrgbd --train-dir /cache/datasets/sunrgbd_Dformer/SUNRGBD -c sunrgbd_swin_large_window12

# eval,54.6
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8848 --use_env main.py --backbone swin_large_window12 --dataset sunrgbd --train-dir /cache/datasets/sunrgbd_Dformer/SUNRGBD --resume sun_54.6.pth.tar --eval

# eval,54.8
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8048 --use_env main.py --backbone swin_large_window12 --dataset sunrgbd --train-dir /cache/datasets/sunrgbd_Dformer/SUNRGBD --resume sun_54.8.pth.tar --eval

# MFNet
# b3
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=8848 --use_env main.py --backbone mit_b3 --dataset mfnet --train-dir /cache/datasets/mfnet -c mfnet_mit_b3
