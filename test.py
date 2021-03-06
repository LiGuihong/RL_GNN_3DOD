import torch
from util.config_util import save_config, save_train_config, \
    load_train_config, load_config
from models.box_encoding import get_box_decoding_fn, get_box_encoding_fn, get_encoding_len
import os
from dataset.kitti_dataset import KittiDataset
from kitty_dataset import DataProvider
from model import *
import numpy as np
import argparse
from util.metrics import recall_precisions, mAP
from tqdm import trange
from tqdm import tqdm
import glob
from torch.utils.tensorboard import SummaryWriter




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training of PointGNN')
    parser.add_argument('train_config_path', type=str,
                       help='Path to train_config')
    parser.add_argument('config_path', type=str,
                       help='Path to config')
    parser.add_argument('--device', type=str, default='cuda:0',
            help="Device for training, cuda or cpu")
    parser.add_argument('--batch_size', type=int, default=1,
            help='Batch size')
    parser.add_argument('--epoches', type=int, default=100,
            help='Training epoches')
    parser.add_argument('--dataset_root_dir', type=str, default='../dataset/kitti/',
                       help='Path to KITTI dataset. Default="../dataset/kitti/"')
    parser.add_argument('--dataset_split_file', type=str,
                        default='',
                       help='Path to KITTI dataset split file.'
                       'Default="DATASET_ROOT_DIR/3DOP_splits'
                       '/train_config["train_dataset"]"')
    parser.add_argument('--k_val', type=int,
                        default=300,
                       help='the number of remained key points')
    parser.add_argument('--vote_idx', type=int,
                        default=1,
                       help='the layer index than prune the graphs')
    parser.add_argument('--resume', type=int,
                        default=0,
                       help='resume from a previous checkpoint or not')
    parser.add_argument('--test_epoch', type=int,
                        default=None,
                       help='the frequency for test')

    args = parser.parse_args()
    epoches = args.epoches
    batch_size = args.batch_size
    device = args.device
    train_config = load_train_config(args.train_config_path)
    DATASET_DIR = args.dataset_root_dir
    config_complete = load_config(args.config_path)
    if 'train' in config_complete:
        config = config_complete['train']
    else:
        config = config_complete

    if args.dataset_split_file == '':
        DATASET_SPLIT_FILE = os.path.join(DATASET_DIR,
            './3DOP_splits/'+train_config['train_dataset'])
    else:
        DATASET_SPLIT_FILE = args.dataset_split_file
    writer = SummaryWriter(log_dir='logs/kp{}_vote{}'.format(args.k_val, args.vote_idx), comment='key points:{}  vote_layer_index:{}'.format(args.k_val, args.vote_idx))
    
    # input function ==============================================================
    train_config['NUM_TEST_SAMPLE']=-1
    train_dataset = KittiDataset(
        os.path.join(DATASET_DIR, 'image/training/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/training/velodyne/'),
        os.path.join(DATASET_DIR, 'calib/training/calib/'),
        os.path.join(DATASET_DIR, 'labels/training/label_2'),
        '/home/guihong/RL_GNN_3DOD/splits/ImageSets/train.txt',
        num_classes=config['num_classes'])
    train_data_provider = DataProvider(train_dataset, train_config, config)

    val_dataset = KittiDataset(
        os.path.join(DATASET_DIR, 'image/training/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/training/velodyne/'),
        os.path.join(DATASET_DIR, 'calib/training/calib/'),
        os.path.join(DATASET_DIR, 'labels/training/label_2'),
        '/home/guihong/RL_GNN_3DOD/splits/ImageSets/val.txt',
        num_classes=config['num_classes'])
    val_data_provider = DataProvider(val_dataset, train_config, config)

    #input_v, vertex_coord_list, keypoint_indices_list, edges_list, \
    #        cls_labels, encoded_boxes, valid_boxes = data_provider.provide_batch([1545, 1546])

    batch = train_data_provider.provide_batch([1545, 1546])
    input_v, vertex_coord_list, keypoint_indices_list, edges_list, \
            cls_labels, encoded_boxes, valid_boxes = batch


    NUM_CLASSES = train_dataset.num_classes
    BOX_ENCODING_LEN = get_encoding_len(config['box_encoding_method'])

    model = MultiLayerFastLocalGraphModelV2(num_classes=NUM_CLASSES,
                box_encoding_len=BOX_ENCODING_LEN, mode='train',
                **config['model_kwargs'])
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    NUM_TRAIN_SAMPLE = train_dataset.num_files
    NUM_VAL_SAMPLE = val_dataset.num_files

    if not args.resume:
        last_epoch = 0
        os.system("mkdir saved_models/kp{}_vote{}".format(args.k_val, args.vote_idx))
    else:
        ckpt_list = glob.glob("saved_models/kp{}_vote{}/model_*".format(args.k_val, args.vote_idx))
        epoch_idx = []
        epoch_idx += [int(ckptname.replace("saved_models/kp{}_vote{}/model_".format(args.k_val, args.vote_idx), '').replace('.pt', '')) for ckptname in ckpt_list]
        last_epoch = np.max(epoch_idx)
        model.load_state_dict(torch.load("saved_models/kp{}_vote{}/model_{}.pt".format(args.k_val, args.vote_idx, last_epoch)))
        model.eval()

        #######################################################
        ######################## Validate #####################

        recalls_list, precisions_list, mAP_list = {}, {}, {}
        for i in range(NUM_CLASSES): recalls_list[i], precisions_list[i], mAP_list[i] = [], [], []
        frame_idx_list = np.random.permutation(NUM_VAL_SAMPLE)
        pbar = tqdm(list(range(0, NUM_VAL_SAMPLE-batch_size+1, batch_size)), desc="start TEST", leave=True)
        model.eval()
        
        all_num = 0
        num_batch = 0
        for batch_idx in pbar:
            #for batch_idx in range(0, NUM_TEST_SAMPLE-batch_size+1, batch_size):
            batch_frame_idx_list = frame_idx_list[batch_idx: batch_idx+batch_size]
            batch = val_data_provider.provide_batch(batch_frame_idx_list)
            input_v, vertex_coord_list, keypoint_indices_list, edges_list, \
                    cls_labels, encoded_boxes, valid_boxes = batch

            new_batch = []
            for item in batch:
                if not isinstance(item, torch.Tensor):
                    item = [x.to(device) for x in item]
                else: item = item.to(device)
                new_batch += [item]
            batch = new_batch
            input_v, vertex_coord_list, keypoint_indices_list, edges_list, \
                    cls_labels, encoded_boxes, valid_boxes = batch

            num_point = model(batch, is_training=False, get_statistic=True)
            all_num = all_num+num_point
            num_batch=num_batch+1
        print(num_point, all_num, num_batch)     
        
        
               