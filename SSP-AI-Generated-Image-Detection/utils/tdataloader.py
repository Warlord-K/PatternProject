import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms, datasets, models
from PIL import Image
import numpy as np
import os
import random as rd
from random import random, choice
from collections import defaultdict
import math
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from io import BytesIO
import cv2


mp = {0: 'imagenet_ai_0508_adm', 1: 'imagenet_ai_0419_biggan', 2: 'imagenet_glide', 3: 'imagenet_midjourney',
      4: 'imagenet_ai_0419_sdv4', 5: 'imagenet_ai_0424_sdv5', 6: 'imagenet_ai_0419_vqdm', 7: 'imagenet_ai_0424_wukong',
      8: 'imagenet_DALLE2'
      }


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def sample_randint(s):
    if len(s) == 1:
        return s[0]
    return rd.randint(s[0], s[1])


def gaussian_blur_gray(img, sigma):
    if len(img.shape) == 3:
        img_blur = np.zeros_like(img)
        for i in range(img.shape[2]):
            img_blur[:, :, i] = gaussian_filter(img[:, :, i], sigma=sigma)
    else:
        img_blur = gaussian_filter(img, sigma=sigma)
    return img_blur


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


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_randint(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def processing(img, opt):
    if opt.aug:
        aug = transforms.Lambda(
            lambda img: data_augment(img, opt)
        )
    else:
        aug = transforms.Lambda(
            lambda img: img
        )

    if opt.isPatch:
        patch_func = transforms.Lambda(
            lambda img: patch_img(img, opt.patch_size, opt.trainsize)) # Assuming patch_img exists
    else:
        patch_func = transforms.Resize((256, 256))

    trans = transforms.Compose([
        aug,
        patch_func,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return trans(img)

def rgb_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


class GenImageDataset(Dataset):
    def __init__(self, image_root, split, opt, csv_path=None, generator_name=None, is_real=None):
        super().__init__()
        self.opt = opt
        self.image_root = image_root
        self.images = []
        self.labels = []

        if csv_path:
            try:
                df = pd.read_csv(csv_path)
                # Assuming csv has 'image_path' and 'label' columns
                # Assuming 'image_path' is relative to image_root
                df['full_path'] = df['image_path'].apply(lambda x: os.path.join(self.image_root, x))

                if split == 'val' and is_real is not None:
                    target_label = 1 if is_real else 0
                    df = df[df['label'] == target_label]

                self.images = df['full_path'].tolist()
                # Ensure labels are integers (0 or 1)
                self.labels = torch.tensor(df['label'].astype(int).tolist())

            except Exception as e:
                print(f"Error loading CSV {csv_path}: {e}")
                # Fallback or raise error? For now, leave lists empty.

        elif generator_name: # Directory mode
            self.root = os.path.join(self.image_root, generator_name, split)
            if split == 'val' and is_real is not None:
                 subfolder = 'nature' if is_real else 'ai'
                 self.img_path = os.path.join(self.root, subfolder)
                 if os.path.isdir(self.img_path):
                     self.img_list = [os.path.join(self.img_path, f) for f in os.listdir(self.img_path)]
                     self.images = self.img_list
                     label_val = 1 if is_real else 0
                     self.labels = torch.full((len(self.images),), label_val)
                 else:
                     print(f"Warning: Directory not found {self.img_path}")

            elif split == 'train' or split == 'test': # Load both nature and ai
                nature_path = os.path.join(self.root, "nature")
                ai_path = os.path.join(self.root, "ai")
                nature_list = []
                ai_list = []
                if os.path.isdir(nature_path):
                    nature_list = [os.path.join(nature_path, f) for f in os.listdir(nature_path)]
                else:
                     print(f"Warning: Directory not found {nature_path}")
                if os.path.isdir(ai_path):
                     ai_list = [os.path.join(ai_path, f) for f in os.listdir(ai_path)]
                else:
                     print(f"Warning: Directory not found {ai_path}")

                self.images = nature_list + ai_list
                self.labels = torch.cat((torch.ones(len(nature_list)), torch.zeros(len(ai_list))))

        else:
             print("Warning: Neither csv_path nor generator_name provided for dataset.")

        if len(self.images) != len(self.labels):
             print(f"Warning: Mismatch between images ({len(self.images)}) and labels ({len(self.labels)}) count.")
             # Attempt to reconcile or raise error? Truncating for now.
             min_len = min(len(self.images), len(self.labels))
             self.images = self.images[:min_len]
             self.labels = self.labels[:min_len]


    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]
        image = rgb_loader(img_path)

        # Handle loading errors
        if image is None:
            print(f"Warning: Replacing failed image at index {index} ({img_path})")
            # Find a valid image to return instead
            valid_index = (index + 1) % len(self.images)
            while rgb_loader(self.images[valid_index]) is None:
                 valid_index = (valid_index + 1) % len(self.images)
                 if valid_index == index: # Avoid infinite loop if all fail
                      # Return a dummy tensor?
                      return torch.zeros((3, 256, 256)), torch.tensor(-1.0) # Indicate error
            image = rgb_loader(self.images[valid_index])
            label = self.labels[valid_index]

        image = processing(image, self.opt)
        # Ensure label is float for potential loss function compatibility
        return image, label.float()

    def __len__(self):
        return len(self.images)


def get_single_loader(opt, image_dir, is_real):
    csv_path = None
    if hasattr(opt, 'val_csv_map') and opt.val_csv_map and image_dir in opt.val_csv_map:
        csv_path = opt.val_csv_map[image_dir]
        print(f"Using CSV for validation: {csv_path}, real={is_real}")
        val_dataset = GenImageDataset(opt.image_root, 'val', opt=opt, csv_path=csv_path, is_real=is_real)
    else:
        print(f"Using Directory for validation: {image_dir}, real={is_real}")
        val_dataset = GenImageDataset(opt.image_root, 'val', opt=opt, generator_name=image_dir, is_real=is_real)

    val_loader = DataLoader(val_dataset, batch_size=opt.val_batchsize,
                            shuffle=False, num_workers=4, pin_memory=True)
    return val_loader, len(val_dataset)


def get_val_loader(opt):
    choices = opt.choices
    loader = []
    for i, choice in enumerate(choices):
        datainfo = dict()
        gen_name = mp.get(i) # Get generator name from map
        if gen_name and (choice == 0 or choice == 1): # Check if choice indicates validation/training
            print("val on:", gen_name)
            datainfo['name'] = gen_name
            datainfo['val_ai_loader'], datainfo['ai_size'] = get_single_loader(
                opt, datainfo['name'], is_real=False)
            datainfo['val_nature_loader'], datainfo['nature_size'] = get_single_loader(
                opt, datainfo['name'], is_real=True)
            loader.append(datainfo)
    return loader


def get_loader(opt):
    image_root = opt.image_root

    if hasattr(opt, 'train_csv') and opt.train_csv:
        print(f"Using Training CSV: {opt.train_csv}")
        train_dataset = GenImageDataset(image_root, 'train', opt=opt, csv_path=opt.train_csv)
    else:
        print("Using Training Directories based on opt.choices")
        datasets = []
        choices = opt.choices
        for i, choice in enumerate(choices):
             gen_name = mp.get(i)
             if gen_name and choice == 1: # choice 1 seems to indicate training inclusion
                  ds = GenImageDataset(image_root, 'train', opt=opt, generator_name=gen_name)
                  datasets.append(ds)
                  print(f"train on: {gen_name}")
        if not datasets:
             print("Warning: No training datasets selected based on opt.choices.")
             return None # Or handle appropriately
        train_dataset = torch.utils.data.ConcatDataset(datasets)

    train_loader = DataLoader(train_dataset, batch_size=opt.batchsize,
                              shuffle=True, num_workers=4, pin_memory=True)
    return train_loader


def get_test_loader(opt):
    image_root = opt.image_root

    if hasattr(opt, 'test_csv') and opt.test_csv:
        print(f"Using Test CSV: {opt.test_csv}")
        test_dataset = GenImageDataset(image_root, 'test', opt=opt, csv_path=opt.test_csv)
    else:
        print("Using Test Directories based on opt.choices")
        datasets = []
        choices = opt.choices
        for i, choice in enumerate(choices):
             gen_name = mp.get(i)
             if gen_name and choice == 2: # choice 2 seems to indicate testing inclusion
                  ds = GenImageDataset(image_root, 'test', opt=opt, generator_name=gen_name)
                  datasets.append(ds)
                  print(f"test on: {gen_name}")
        if not datasets:
             print("Warning: No test datasets selected based on opt.choices.")
             return None # Or handle appropriately
        test_dataset = torch.utils.data.ConcatDataset(datasets)

    test_loader = DataLoader(test_dataset, batch_size=opt.batchsize,
                             shuffle=False, num_workers=4, pin_memory=True) # Usually no shuffle for test
    return test_loader
