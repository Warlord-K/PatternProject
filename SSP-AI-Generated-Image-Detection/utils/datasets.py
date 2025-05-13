import os
import pandas as pd
from io import BytesIO
from random import choice, random

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from scipy.ndimage import gaussian_filter
from torch.utils.data.sampler import WeightedRandomSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True

def handle_dataset(opt):
    return GenImageDataset(opt.image_root, 'train', opt=opt)

class GenImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode, opt=None):
        self.root_dir = root_dir
        self.mode = mode
        self.opt = opt
        self.transform = get_transform(opt)
        
        self.images = []
        self.labels = []
        
        for class_folder in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_folder)
            if not os.path.isdir(class_path):
                continue
                
            label = 1 if class_folder.lower() == 'real' else 0
            
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(label)
        
        self.targets = self.labels  # For compatibility with balanced sampler
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.targets = self.data_frame['label'].tolist()  # For compatibility with balanced sampler
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1])
        image = Image.open(img_name).convert('RGB')
        label = self.data_frame.iloc[idx, 2]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)


def get_transform(cfg):
    if cfg is None:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    transform_list = []

    # Resize transform
    if hasattr(cfg, 'loadSize'):
        transform_list.append(transforms.Resize((cfg.loadSize, cfg.loadSize)))
    else:
        transform_list.append(transforms.Resize((256, 256)))
    
    # Blur and JPEG augmentation
    if hasattr(cfg, 'blur_prob') and cfg.blur_prob > 0 or hasattr(cfg, 'jpg_prob') and cfg.jpg_prob > 0:
        transform_list.append(transforms.Lambda(lambda img: blur_jpg_augment(img, cfg)))
    
    # Crop transform
    if hasattr(cfg, 'cropSize'):
        if hasattr(cfg, 'isTrain') and cfg.isTrain:
            transform_list.append(transforms.RandomCrop(cfg.cropSize))
        else:
            transform_list.append(transforms.CenterCrop(cfg.cropSize))
    
    # Flip transform
    if hasattr(cfg, 'isTrain') and cfg.isTrain and hasattr(cfg, 'aug_flip') and cfg.aug_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    # Convert to tensor and normalize
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def blur_jpg_augment(img, cfg):
    img = np.array(img)
    
    if hasattr(cfg, 'blur_prob') and random() < cfg.blur_prob:
        sig = 1.0  # Default value
        if hasattr(cfg, 'blur_sig'):
            if isinstance(cfg.blur_sig, list):
                if len(cfg.blur_sig) == 1:
                    sig = cfg.blur_sig[0]
                elif len(cfg.blur_sig) == 2:
                    sig = random() * (cfg.blur_sig[1] - cfg.blur_sig[0]) + cfg.blur_sig[0]
            else:
                sig = cfg.blur_sig
        gaussian_blur(img, sig)

    if hasattr(cfg, 'jpg_prob') and random() < cfg.jpg_prob:
        method = "cv2"
        qual = 80
        
        if hasattr(cfg, 'jpg_method'):
            if isinstance(cfg.jpg_method, list):
                method = choice(cfg.jpg_method)
            else:
                method = cfg.jpg_method
                
        if hasattr(cfg, 'jpg_qual'):
            if isinstance(cfg.jpg_qual, list):
                if len(cfg.jpg_qual) == 1:
                    qual = cfg.jpg_qual[0]
                elif len(cfg.jpg_qual) == 2:
                    qual = choice(range(cfg.jpg_qual[0], cfg.jpg_qual[1] + 1))
            else:
                qual = cfg.jpg_qual
                
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


def get_dataset(cfg):
    if hasattr(cfg, 'datasets') and cfg.datasets:
        dset_lst = []
        for dataset in cfg.datasets:
            csv_path = os.path.join(cfg.dataset_root, dataset + '.csv')
            root_dir = cfg.dataset_root
            transform = get_transform(cfg)
            dset = CSVDataset(csv_path, root_dir, transform)
            dset_lst.append(dset)
        
        if len(dset_lst) == 1:
            return dset_lst[0]
        return torch.utils.data.ConcatDataset(dset_lst)
    else:
        return GenImageDataset(cfg.image_root, 'train', opt=cfg)


def get_val_loader(cfg):
    dataset = get_dataset(cfg)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batchsize if hasattr(cfg, 'batchsize') else 32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


def create_dataloader(cfg):
    shuffle = True
    if hasattr(cfg, 'serial_batches') and cfg.serial_batches:
        shuffle = False
        
    dataset = get_dataset(cfg)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batchsize if hasattr(cfg, 'batchsize') else 32,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )