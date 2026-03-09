import argparse
import random
from argparse import ArgumentParser

import os
import numpy as np
import random
import datetime
import PIL.Image as Image

import logging
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import autocast, GradScaler

from adamp import AdamP

from itertools import cycle
import pyiqa

from torch.utils.tensorboard import SummaryWriter


from collections import OrderedDict
try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


current_time = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")


def get_available_devices(n_gpu):
    sys_gpu = torch.cuda.device_count()
    if sys_gpu == 0:
        print('No GPUs detected, using the CPU')
        n_gpu = 0
    elif n_gpu > sys_gpu:
        print(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
        n_gpu = sys_gpu
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    available_gpus = list(range(n_gpu))
    return device, available_gpus

def get_args():
    parser = argparse.ArgumentParser(description='Evaliation')

    ### >>> [data] set data info.
    parser.add_argument('--database', default='TIUIED', type=str, help='database name')
    parser.add_argument('--crop_size', type=int, default=256, help='input image size for training (default: 256)')

    ### >>> [model] set model info.
    parser.add_argument('--model_E', default='DTIUIE', type=str, help='model name')
    parser.add_argument('--pretrained', default=None, type=str, help='path to latest checkpoint (default: None)')

    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('-g', '--gpus', type=int, default=1, metavar='N')

    parser.add_argument('--draw_images', action='store_true', default=False, help='flag whether to draw images')
    parser.add_argument('--save_dir', default='output_images', type=str, help='path to save images')

    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()
    args.local_rank = -1

    device, available_gpus = get_available_devices(args.gpus)

    # >>> [model] set model info.
    # >>> [model] set model info.
    if  args.model_E == 'TIUIED':
        from models.enhancement.model_dtiuie import TIUIED
        model_E = TwoBranchNetwork_SA()
        model_E = model_E.to(device)
        args.pretrained = './checkpoints/dtiuie_ckpt.pth'
    

    from models.segmentation.vgg_unet import VGG16Unet, VGG13Unet
    from models.segmentation.resnet_unet import ResNet50Unet

    model_S = VGG16Unet(n_channels=3, n_classes=8, pretrained=True)
    model_S = model_S.to(device)


    from dataloader_seg import Tested_Set
    dataset_dir = './dataset/test/images'
    args.classes = 8
    dataset = Tested_Set(file_path = dataset_dir, status='valid', augmentation=False, angle=0, size_h=args.crop_size, size_w=args.crop_size, hflip_p=0)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, drop_last=True, batch_size=1, num_workers=8, pin_memory=True)

    if args.pretrained:
        pre_trained_state_dict = torch.load(args.pretrained)
        model_E.load_state_dict(pre_trained_state_dict['model_E'])
        model_S.load_state_dict(pre_trained_state_dict['model_S'])
    model_E.to(device=device), model_S.to(device=device)

    model_E.eval(), model_S.eval()
    num_val_batches = len(dataloader)

    epoch_valid_psnr = []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', ncols=150, leave=True):

            images = batch['image'].to(device)
            _, _, original_h, original_w = images.shape
            new_h = (original_h + 31) // 32 * 32 
            new_w = (original_w + 31) // 32 * 32 
            if original_h * original_w > 1088 * 1920:
                new_h = 1088
                new_w = 1920
            images = F.interpolate(images, size=(new_h, new_w), mode='bicubic', align_corners=False)
            original_h, original_w = 256, 256
            images = F.interpolate(images, size=(256, 256), mode='bicubic', align_corners=False)
            images_names = batch['image_name']

            # Generate output
            # Generate output
            if 'sa' in args.model_E:
                _, feat_raw = model_S(images, return_feats=True)
                outputs = model_E(images, feat_raw)
            else:
                outputs = model_E(images)


            if args.draw_images:
                for i, (output, image_name) in enumerate(zip(outputs, images_names)):
                    output = output.unsqueeze(0)
                    output = F.interpolate(output, size=(original_h, original_w), mode='bicubic', align_corners=False)
                    output = output.squeeze(0)

                    output = output.permute(1, 2, 0).cpu().numpy()
                    output = (output * 255).astype(np.uint8)
                    output = Image.fromarray(output)
                    
                    output.save(f'./outputs/DTIUIE/{os.path.basename(image_name)}')


    

