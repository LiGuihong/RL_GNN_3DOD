CUDA_VISIBLE_DEVICES=1 python3 trainrl.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config --dataset_root_dir /mnt/17b83cc4-8721-4108-b173-4fa1677ba5df/dataset/kitti --k_val=-1 &
CUDA_VISIBLE_DEVICES=2 python3 trainrl.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config --dataset_root_dir /mnt/17b83cc4-8721-4108-b173-4fa1677ba5df/dataset/kitti --k_val=256 --vote_idx=1 &
CUDA_VISIBLE_DEVICES=3 python3 trainrl.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config --dataset_root_dir /mnt/17b83cc4-8721-4108-b173-4fa1677ba5df/dataset/kitti --k_val=512 --vote_idx=1 &
wait

CUDA_VISIBLE_DEVICES=1 python3 trainrl.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config --dataset_root_dir /mnt/17b83cc4-8721-4108-b173-4fa1677ba5df/dataset/kitti --k_val=1024 --vote_idx=1 &
CUDA_VISIBLE_DEVICES=2 python3 trainrl.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config --dataset_root_dir /mnt/17b83cc4-8721-4108-b173-4fa1677ba5df/dataset/kitti --k_val=512 --vote_idx=0 &
CUDA_VISIBLE_DEVICES=3 python3 trainrl.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config --dataset_root_dir /mnt/17b83cc4-8721-4108-b173-4fa1677ba5df/dataset/kitti --k_val=1024 --vote_idx=0 &
wait

CUDA_VISIBLE_DEVICES=3 python3 trainrl.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config --dataset_root_dir /mnt/17b83cc4-8721-4108-b173-4fa1677ba5df/dataset/kitti --k_val=256 --vote_idx=0 &
CUDA_VISIBLE_DEVICES=1 python3 trainrl.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config --dataset_root_dir /mnt/17b83cc4-8721-4108-b173-4fa1677ba5df/dataset/kitti --k_val=128 --vote_idx=1 &
CUDA_VISIBLE_DEVICES=2 python3 trainrl.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config --dataset_root_dir /mnt/17b83cc4-8721-4108-b173-4fa1677ba5df/dataset/kitti --k_val=128 --vote_idx=0 &
wait

#python3 kitty_dataset.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config --dataset_root_dir data
#python3 train.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config --dataset_root_dir data

CUDA_VISIBLE_DEVICES=3 python3 trainrl.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config --dataset_root_dir /mnt/17b83cc4-8721-4108-b173-4fa1677ba5df/dataset/kitti --k_val=512 --vote_idx=1 --resume=1

CUDA_VISIBLE_DEVICES=1 python3 test.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config \
--dataset_root_dir /mnt/4889cd91-2b08-4919-8d37-9ca1657e954e/datasets/kitti_mirror \
--k_val=-1 --vote_idx=-1  --resume=1 --test_epoch=10


CUDA_VISIBLE_DEVICES=0 python3 rlagent.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config \
 --dataset_root_dir /mnt/4889cd91-2b08-4919-8d37-9ca1657e954e/datasets/kitti_mirror \
 --k_val=-1 --vote_idx=-1 --resume=1 --test_epoch=2


