import argparse
import os
import sys
from utils.config import CONFIGCLASS

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='gpu ids to use')
        self.parser.add_argument('--exp_name', type=str, default='experiment', help='experiment name')
        self.parser.add_argument('--image_root', type=str, default="/Users/yatharthgupta/vscode/PatternProject/data/4", help='path to image data')
        self.parser.add_argument('--save_path', type=str, default='./checkpoint/', help='models are saved here')
        self.parser.add_argument('--load', type=str, default=None, help='load model from a .pth file')
        self.parser.add_argument('--gpu_id', type=str, default='0', help='gpu id: 0, 1, 2')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
        self.parser.add_argument('--batchsize', type=int, default=32, help='batch size')
        self.parser.add_argument('--epoch', type=int, default=300, help='number of epochs')
        self.parser.add_argument('--dataset_root', type=str, default="/Users/yatharthgupta/vscode/PatternProject/data/4", help='dataset root directory')
        self.parser.add_argument('--datasets', nargs='+', default=['train'], help='datasets to use')
        
        # Add augmentation arguments
        self.parser.add_argument('--blur_prob', type=float, default=0.0, help='probability of applying blur')
        self.parser.add_argument('--jpg_prob', type=float, default=0.0, help='probability of applying jpeg compression')
        self.parser.add_argument('--CropSize', type=int, default=224, help='crop size for training')
        self.parser.add_argument('--trainsize', type=int, default=256, help='size to resize images before cropping')
        
    def parse(self):
        args = self.parser.parse_args()
        
        # Update configuration with command line arguments
        if args.dataset_root is not None:
            CONFIGCLASS.dataset_root = args.dataset_root
        if args.image_root is not None:
            CONFIGCLASS.image_root = args.image_root
            if CONFIGCLASS.dataset_root is None:
                CONFIGCLASS.dataset_root = args.image_root
                
        CONFIGCLASS.save_path = args.save_path
        CONFIGCLASS.load = args.load
        CONFIGCLASS.gpu_id = args.gpu_id
        CONFIGCLASS.gpus = [int(args.gpu_id)] if args.gpu_id else args.gpus
        CONFIGCLASS.lr = args.lr
        CONFIGCLASS.batchsize = args.batchsize
        CONFIGCLASS.batch_size = args.batchsize  # For compatibility
        CONFIGCLASS.nepoch = args.epoch
        CONFIGCLASS.epoch = args.epoch  # For compatibility
        CONFIGCLASS.datasets = args.datasets
        
        # Update augmentation parameters
        CONFIGCLASS.blur_prob = args.blur_prob
        CONFIGCLASS.jpg_prob = args.jpg_prob
        CONFIGCLASS.cropSize = args.CropSize
        CONFIGCLASS.loadSize = args.trainsize
        
        # Setup environment
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(gpu) for gpu in CONFIGCLASS.gpus])
        
        return CONFIGCLASS