
```
conda install pytorch torchvision

```
Install torch-scatter according to your pytorch version following instructions in this url: https://github.com/rusty1s/pytorch_scatter

To install other dependencies: 
```
pip3 install --user opencv-python
pip3 install --user open3d-python==0.7.0.0
pip3 install --user scikit-learn
pip3 install --user tqdm
pip3 install --user shapely
```

### KITTI Dataset

We use the KITTI 3D Object Detection dataset. Please download the dataset from the KITTI website and also download the 3DOP train/val split [here](https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz). We provide extra split files for seperated classes in [splits/](splits). We recommand the following file structure:

```
ssh to lambdax.ece.utexas.edu
DATASET_ROOT_DIR: /mnt/17b83cc4-8721-4108-b173-4fa1677ba5df/dataset/kitti 
```
    DATASET_ROOT_DIR
    ├── image                    #  Left color images
    │   ├── training
    |   |   └── image_2            
    │   └── testing
    |       └── image_2 
    ├── velodyne                 # Velodyne point cloud files
    │   ├── training
    |   |   └── velodyne            
    │   └── testing
    |       └── velodyne 
    ├── calib                    # Calibration files
    │   ├── training
    |   |   └──calib            
    │   └── testing
    |       └── calib 
    ├── labels                   # Training labels
    │   └── training
    |       └── label_2
    └── 3DOP_splits              # split files.
        ├── train.txt
        ├── train_car.txt
        └── ...

### Download Point-GNN
```
git clone 
```

### Prepare work dir
```
mkdir logs
mkdir saved_models
mkdir figs
```

### Backbone Training
```
python3 trainrl.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config \
  --dataset_root_dir /mnt/17b83cc4-8721-4108-b173-4fa1677ba5df/dataset/kitti --k_val=-1 --vote_idx=-1 
```

### RL Training
```
python3 combining.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config \
  --dataset_root_dir /mnt/17b83cc4-8721-4108-b173-4fa1677ba5df/dataset/kitti --k_val=-1 --vote_idx=-1 
```

