# # NYU
# # b3
# CUDA_VISIBLE_DEVICES=4,5,6 python -m torch.distributed.launch --nproc_per_node=3 --master_port 8848 --use_env main.py --backbone mit_b3 --dataset nyudv2 --eval --resume /home/ma-user/work/jiading/Public_Prepare/Ready/checkpoints/nyu/new/mit-b3.pth.tar

# # b5
# CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --master_port 8748 --use_env main.py --backbone mit_b5 --dataset nyudv2 --eval --resume /home/ma-user/work/jiading/Public_Prepare/Ready/checkpoints/nyu/new/mit-b5.pth.tar --dpr 0.35

# # swin_tiny
# CUDA_VISIBLE_DEVICES=3,7,1 python -m torch.distributed.launch --nproc_per_node=3 --master_port 8648 --use_env main.py --backbone swin_tiny --dataset nyudv2 --eval --resume /home/ma-user/work/jiading/Public_Prepare/Ready/checkpoints/nyu/new/swin-tiny.pth.tar --dpr 0.2

# # swin_small
# CUDA_VISIBLE_DEVICES=4,5,6 python -m torch.distributed.launch --nproc_per_node=3 --master_port 8548 --use_env main.py --backbone swin_small --dataset nyudv2 --eval --resume /home/ma-user/work/jiading/Public_Prepare/Ready/checkpoints/nyu/new/swin-small.pth.tar

# # swin_large
# CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --master_port 8448 --use_env main.py --backbone swin_large --dataset nyudv2 --eval --resume /home/ma-user/work/jiading/Public_Prepare/Ready/checkpoints/nyu/new/swin-large-224.pth.tar

# # swin_large_window12
# CUDA_VISIBLE_DEVICES=3,7,1 python -m torch.distributed.launch --nproc_per_node=3 --master_port 8348 --use_env main.py --backbone swin_large_window12 --dataset nyudv2 --eval --resume /home/ma-user/work/jiading/Public_Prepare/Ready/checkpoints/nyu/new/swin-large-384.pth.tar --dpr 0.2

# swin-large-384+dpr0.15+100-200-100+FineTune from SUN 300eps
# CUDA_VISIBLE_DEVICES=4,5,6 python -m torch.distributed.launch --nproc_per_node=3 --master_port 8248 --use_env main.py --backbone swin_large_window12 --dataset nyudv2 --eval --resume /home/ma-user/work/jiading/Public_Prepare/Ready/checkpoints/nyu/new/finetune-swin-large-384.pth.tar



# # SUN
# # b3
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8848 --use_env main.py --backbone mit_b3 --dataset sunrgbd --train-dir /cache/datasets/sunrgbd_Dformer/SUNRGBD --eval --resume /home/ma-user/work/jiading/Public_Prepare/Ready/checkpoints/sun/raw/mit-b3.pth.tar

# # b5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8748 --use_env main.py --backbone mit_b5 --dataset sunrgbd --train-dir /cache/datasets/sunrgbd_Dformer/SUNRGBD --eval --resume /home/ma-user/work/jiading/Public_Prepare/Ready/checkpoints/sun/raw/mit-b5.pth.tar --weight_decay 0.05

# # swin_tiny
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8648 --use_env main.py --backbone swin_tiny --dataset sunrgbd --train-dir /cache/datasets/sunrgbd_Dformer/SUNRGBD --eval --resume /home/ma-user/work/jiading/Public_Prepare/Ready/checkpoints/sun/raw/swin-tiny.pth.tar

# # swin_large_window12
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8548 --use_env main.py --backbone swin_large_window12 --dataset sunrgbd --train-dir /cache/datasets/sunrgbd_Dformer/SUNRGBD --eval --resume /home/ma-user/work/jiading/Public_Prepare/Ready/checkpoints/sun/raw/swin-large-384.pth.tar
