export PYTHONPATH="/home/ma-user/work/jiading/Public_Prepare/Ready/Code/Deliver/TokenFusion_CMNext_enable_multi_modals_enable_sun"

# b2,rgb+d  | 66.4 
python -m torch.distributed.launch --master_port 2255 --nproc_per_node=8 --use_env tools/train_mm.py --cfg configs/deliver_rgbd_8cards_2e-4.yaml \
--drop_path_rate 0.4

# b2,rgb+e  | 58.5 |
python -m torch.distributed.launch --master_port 2245 --nproc_per_node=8 --use_env tools/train_mm.py --cfg configs/deliver_rgbe_8cards_2e-4.yaml \
--drop_path_rate 0.4

# b2,rgb+l  | 58.6 |
python -m torch.distributed.launch --master_port 2235 --nproc_per_node=8 --use_env tools/train_mm.py --cfg configs/deliver_rgbl_8cards_2e-4.yaml \
--drop_path_rate 0.4

# b2,rgb+d+e+l  | 66.9 |
python -m torch.distributed.launch --master_port 2225 --nproc_per_node=8 --use_env tools/train_mm.py --cfg configs/deliver_rgbdel_8cards_2e-4.yaml \
--drop_path_rate 0.2