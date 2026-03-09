from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as F
from torchvision.transforms.functional import resize, rotate, crop, hflip, vflip, to_tensor, normalize
from torchvision.transforms import InterpolationMode
from PIL import Image
from cProfile import label
import os


import numpy as np
import pandas as pd
import torch 

import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler 

from PIL import Image


from skimage.color import rgb2hsv, rgb2lab
from os import listdir
from os.path import splitext, isfile, join
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def rotate(img, rotate_index):
    '''
    :return: 8 version of rotating image
    '''
    if rotate_index == 0:
        return img
    if rotate_index==1:
        return img.rotate(90)
    if rotate_index==2:
        return img.rotate(180)
    if rotate_index==3:
        return img.rotate(270)
    if rotate_index==4:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index==5:
        return img.rotate(90).transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index==6:
        return img.rotate(180).transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index==7:
        return img.rotate(270).transpose(Image.FLIP_TOP_BOTTOM)


def data_loader(dataset_name, batch_size=8, size=256):

    if dataset_name == 'SUIM':
        train_path = "./TaskFriendly_UIE/underwater_data/trainval/"
        valid_path = "./TaskFriendly_UIE/underwater_data/test/"
        

        train_dataset = Paired_Set(file_path = train_path, status='train', augmentation=True,
                               angle=2, size_h=size, size_w=size, hflip_p=0.5)
        valid_dataset = Paired_Set(file_path = valid_path, status='valid', augmentation=False,
                               angle=0, size_h=size, size_w=size, hflip_p=0)

    else:
        print("!!! Dataset " + dataset_name + " name not matched.")
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, sampler=None)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=4, shuffle=False, num_workers=8, sampler=None)

    return train_loader, valid_loader


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)
    
def unique_mask_values(mask_dir):
    unique_values = []
    for mask_file in mask_dir:
        mask = np.asarray(load_image(mask_file))
        if mask.ndim == 2:
            unique_values.append(np.unique(mask))
        elif mask.ndim == 3:
            mask = mask.reshape(-1, mask.shape[-1])
            unique_values.append(np.unique(mask, axis=0))
        else:
            raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')
    # Flatten the list of unique values and return the consolidated unique values
    return np.unique(np.concatenate(unique_values), axis=0)

class Paired_Set(torch.utils.data.Dataset):
    def __init__(self, file_path, status, augmentation, angle, size_h, size_w, hflip_p):
        super(Paired_Set).__init__()
        self.status = status
        self.augment = augmentation
        self.angle = angle
        self.size_h = size_h
        self.size_w = size_w
        self.hflip_p = hflip_p

        self.images_paths = sorted(self.load_data(file_path + 'images'))
        self.images_gt_paths = sorted(self.load_data(file_path + 'reference'))
        self.images_mask_paths = sorted(self.load_data(file_path + 'masks_png'))


        print(f'Creating dataset with {len(self.images_paths)} examples')
        print('Scanning mask files to determine unique values')

        unique = unique_mask_values(mask_dir=self.images_mask_paths)

        self.mask_values = list(sorted(unique.tolist()))
        
        print(f'Unique mask values: {self.mask_values}')

    def __getitem__(self, index):
        random = np.random.rand(1)

        image_name = self.images_paths[index]
        image = Image.open(image_name).convert('RGB')
        image = self.transform(image, self.status, False, self.size_w, self.size_h, random)

        image_gt_name = self.images_gt_paths[index]
        image_gt = Image.open(image_gt_name).convert('RGB')
        image_gt = self.transform(image_gt, self.status, False, self.size_w, self.size_h, random)

        image_mask_name = self.images_mask_paths[index]
        image_mask = Image.open(image_mask_name).convert('L')
        image_mask = self.transform(image_mask, self.status, True, self.size_w, self.size_h, random)

        return {'image': F.to_tensor(image.copy()),
                 'image_gt': F.to_tensor(image_gt.copy()),
                 'image_mask': torch.as_tensor(np.asarray(image_mask).copy()).long(),
                 'image_name': image_name}
    
    def __len__(self):
        return len(self.images_paths)

    def load_data(self, file_path):
        image_names = []
        assert os.path.isdir(file_path), '%s is not a valid directory' % file_path
        for root, _, fnames in sorted(os.walk(file_path)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    image_names.append(path)
        print(">>> DATALOADER >>> " + self.status + " data size is :" + str(len(image_names)))
        return image_names
    
    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def transform(self, image, status, is_mask, size_w, size_h, random):
        if status == 'train' and self.augment:  # data augmentation
            w, h = image.size
            ## >> 1. resize a bigger image.
            if is_mask:
                image = F.resize(image, (size_w, size_h), interpolation=InterpolationMode.NEAREST)
                # image = F.hflip(image) if random < self.hflip_p else image
            else:
                image = F.resize(image, (size_w, size_h), interpolation=InterpolationMode.BILINEAR)
                # image = F.hflip(image) if random < self.hflip_p else image
                #image = F.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            
        else:
            if is_mask:
                image = F.resize(image, (size_w, size_h), interpolation=InterpolationMode.NEAREST)
            else:
                image = F.resize(image, (size_w, size_h), interpolation=InterpolationMode.BILINEAR)
                #image = F.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        return image



class Tested_Set(Paired_Set):
    def __init__(self, file_path, status, augmentation, angle, size_h, size_w, hflip_p):
        super(Paired_Set).__init__()
        self.status = status
        self.augment = augmentation
        self.angle = angle
        self.size_h = size_h
        self.size_w = size_w
        self.hflip_p = hflip_p

        self.images_paths = sorted(self.load_data(file_path))

        print(f'Creating dataset with {len(self.images_paths)} examples')

    def __getitem__(self, index):
        image_name = self.images_paths[index]
        image = Image.open(image_name).convert('RGB')
        # image = self.transform(image, self.status, False, self.size_w, self.size_h)

        return {'image': F.to_tensor(image.copy()),
                 'image_name': image_name}

    