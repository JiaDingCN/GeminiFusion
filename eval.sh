export PYTHONPATH="/home/ma-user/work/jiading/Public_Prepare/Ready/Code/Deliver/TokenFusion_CMNext_enable_multi_modals_enable_sun"

# b2,rgb+d  | 66.4 
CUDA_VISIBLE_DEVICES=3 python tools/val_mm.py --cfg configs/deliver_rgbd_8cards_2e-4.yaml 

# b2,rgb+e  | 58.5 |
CUDA_VISIBLE_DEVICES=2 python tools/val_mm.py --cfg configs/deliver_rgbe_8cards_2e-4.yaml 
# b2,rgb+l  | 58.6 |
CUDA_VISIBLE_DEVICES=1 python tools/val_mm.py --cfg configs/deliver_rgbl_8cards_2e-4.yaml 

# b2,rgb+d+e+l  | 66.9 |
CUDA_VISIBLE_DEVICES=0 python tools/val_mm.py --cfg configs/deliver_rgbdel_8cards_2e-4.yaml 