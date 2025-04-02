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

from utils.config import CONFIGCLASS

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
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
            
        return image, label


def get_transform(cfg: CONFIGCLASS):
    """Create a transform pipeline based on config"""
    identity_transform = transforms.Lambda(lambda img: img)
    transform_list = []

    # Resize transform
    if cfg.isTrain or cfg.aug_resize:
        transform_list.append(transforms.Lambda(lambda img: custom_resize(img, cfg)))
    
    # Blur and JPEG augmentation
    transform_list.append(transforms.Lambda(lambda img: blur_jpg_augment(img, cfg)))
    
    # Crop transform
    if cfg.isTrain:
        transform_list.append(transforms.RandomCrop(cfg.cropSize))
    elif cfg.aug_crop:
        transform_list.append(transforms.CenterCrop(cfg.cropSize))
    
    # Flip transform
    if cfg.isTrain and cfg.aug_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    # Convert to tensor and normalize
    transform_list.append(transforms.ToTensor())
    if cfg.aug_norm:
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)


def blur_jpg_augment(img: Image.Image, cfg: CONFIGCLASS):
    img: np.ndarray = np.array(img)
    if cfg.isTrain:
        if random() < cfg.blur_prob:
            sig = sample_continuous(cfg.blur_sig)
            gaussian_blur(img, sig)

        if random() < cfg.jpg_prob:
            method = sample_discrete(cfg.jpg_method)
            qual = sample_discrete(cfg.jpg_qual)
            img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s: list):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s: list):
    return s[0] if len(s) == 1 else choice(s)


def gaussian_blur(img: np.ndarray, sigma: float):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img: np.ndarray, compress_val: int) -> np.ndarray:
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img: np.ndarray, compress_val: int):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}


def jpeg_from_key(img: np.ndarray, compress_val: int, key: str) -> np.ndarray:
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}


def custom_resize(img: Image.Image, cfg: CONFIGCLASS) -> Image.Image:
    interp = sample_discrete(cfg.rz_interp)
    return TF.resize(img, cfg.loadSize, interpolation=rz_dict[interp])


def get_dataset(cfg: CONFIGCLASS):
    """Create datasets from CSV files"""
    dset_lst = []
    for dataset in cfg.datasets:
        # Assuming dataset is now a CSV file path
        csv_path = os.path.join(cfg.dataset_root, dataset + '.csv')
        root_dir = cfg.dataset_root  # Base directory containing the images
        transform = get_transform(cfg)
        dset = CSVDataset(csv_path, root_dir, transform)
        dset_lst.append(dset)
    
    if len(dset_lst) == 1:
        return dset_lst[0]
    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset):
    """Get a balanced sampler for training"""
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        targets = []
        for d in dataset.datasets:
            targets.extend(d.targets)
    else:
        targets = dataset.targets

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))


def create_dataloader(cfg: CONFIGCLASS):
    shuffle = not cfg.serial_batches if (cfg.isTrain and not cfg.class_bal) else False
    dataset = get_dataset(cfg)
    sampler = get_bal_sampler(dataset) if cfg.class_bal else None

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
    )