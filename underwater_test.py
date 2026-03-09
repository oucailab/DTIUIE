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
    parser.add_argument('--database', default='TF_SUIM', type=str, help='database name (default: LIVE)')
    parser.add_argument('--crop_size', type=int, default=256, help='input image size for training (default: 256)')

    ### >>> [model] set model info.
    parser.add_argument('--model_E', default='my_UIE_mpvit_sa_mix', type=str, help='model name ((Wa)DIQaM-FR/NR, default: WaDIQaM-FR)')
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
    if args.model_E == 'my_UIE_mpvit_sa':
        from models.enhancement.model_E_UNet_mpvit import TwoBranchNetwork_SA
        model_E = TwoBranchNetwork_SA()
        model_E = model_E.to(device)
        args.pretrained = '/home/catchacat/workspace/Underwater_enhancement/my_TFUIE_new/checkpoints/my_UIE_mpvit_sa_vgg/my_UIE_mpvit_sa-my_UIE-SUIM-lr=0.0001-wd=0.001-bs=4-03-20-2025-16-24-05-best'
    elif args.model_E == 'my_UIE_mpvit_sa_mix':
        from models.enhancement.model_E_UNet_mpvit_mix_3mix import TwoBranchNetwork_SA
        model_E = TwoBranchNetwork_SA()
        model_E = model_E.to(device)
        args.pretrained = '/home/catchacat/workspace/Underwater_enhancement/my_TFUIE_new/checkpoints/my_UIE_mpvit_sa_mix_vgg/my_UIE_mpvit_sa_mix-my_UIE-SUIM-lr=0.0001-wd=0.001-bs=4-03-20-2025-16-24-19-best'
    
    
    
    elif args.model_E == 'my_UIE_mpvit_sa_mix_cross':
        from models.enhancement.model_E_UNet_mpvit_cross import TwoBranchNetwork_SA
        model_E = TwoBranchNetwork_SA()
        model_E = model_E.to(device)    
        args.pretrained = '/home/catchacat/workspace/Underwater_enhancement/my_TFUIE_new/checkpoints/my_UIE_mpvit_sa_mix_vgg/my_UIE_mpvit_sa_mix-my_UIE-SUIM-lr=0.0001-wd=0.001-bs=4-03-27-2025-12-09-26-best'
    elif args.model_E == 'my_UIE_mpvit_sa_mix_single':
        from models.enhancement.model_E_UNet_mpvit_mix_single import TwoBranchNetwork_SA
        model_E = TwoBranchNetwork_SA()
        model_E = model_E.to(device)    
        args.pretrained = '/home/catchacat/workspace/Underwater_enhancement/my_TFUIE_new/checkpoints/my_UIE_mpvit_sa_vgg/my_UIE_mpvit_sa-my_UIE-SUIM-lr=0.0001-wd=0.001-bs=4-03-13-2025-10-17-32-best'
    elif args.model_E == 'my_UIE_ab_sa_vgg13':
        from models.enhancement.model_E_UNet_mpvit_mix import TwoBranchNetwork_SA
        model_E = TwoBranchNetwork_SA()
        model_E = model_E.to(device)
        args.pretrained = '/home/catchacat/workspace/Underwater_enhancement/my_TFUIE_new/checkpoints/my_UIE_ab_sa_vgg13_vgg/my_UIE_ab_sa_vgg13-my_UIE-SUIM-lr=0.0001-wd=0.001-bs=4-06-14-2025-14-31-39-best_ssim.pth'
    elif args.model_E == 'my_UIE_ab_sa_res50':
        from models.enhancement.model_E_UNet_mpvit_mix_res import TwoBranchNetwork_SA
        model_E = TwoBranchNetwork_SA()
        model_E = model_E.to(device)    
        args.pretrained = '/home/catchacat/workspace/Underwater_enhancement/my_TFUIE_new/checkpoints/my_UIE_ab_sa_res50_vgg/my_UIE_ab_sa_res50-my_UIE-SUIM-lr=0.0001-wd=0.001-bs=4-06-14-2025-15-39-37-best_ssim.pth'

    elif args.model_E == 'my_UIE_mpvit_sa_mix_suime':
        from models.enhancement.model_E_UNet_mpvit_mix_3mix import TwoBranchNetwork_SA
        model_E = TwoBranchNetwork_SA()
        model_E = model_E.to(device)
        args.pretrained = '/home/catchacat/workspace/Underwater_enhancement/my_TFUIE_new/checkpoints/my_UIE_mpvit_sa_mix_vgg/my_UIE_mpvit_sa_mix-my_UIE-SUIM_E-lr=0.0001-wd=0.001-bs=4-03-27-2025-00-33-21-best'

    elif args.model_E == 'my_UIE_mpvit_sa_mix_no_mix_loss':
        from models.enhancement.model_E_UNet_mpvit_mix import TwoBranchNetwork_SA
        model_E = TwoBranchNetwork_SA()
        model_E = model_E.to(device)
        args.pretrained = '/home/catchacat/workspace/Underwater_enhancement/my_TFUIE_new/checkpoints/my_UIE_mpvit_sa_mix_vgg/my_UIE_mpvit_sa_mix-my_UIE-SUIM-lr=0.0001-wd=0.001-bs=4-01-05-2026-12-39-58-best_ssim.pth'
    elif args.model_E == 'my_UIE_mpvit_sa_mix_pixel_mix_loss':
        from models.enhancement.model_E_UNet_mpvit_mix import TwoBranchNetwork_SA
        model_E = TwoBranchNetwork_SA()
        model_E = model_E.to(device)
        args.pretrained = '/home/catchacat/workspace/Underwater_enhancement/my_TFUIE_new/checkpoints/my_UIE_mpvit_sa_mix_vgg/my_UIE_mpvit_sa_mix-my_UIE-SUIM-lr=0.0001-wd=0.001-bs=4-01-05-2026-13-25-51-best_ssim.pth'

    elif args.model_E == 'my_UIE_mpvit_sa_mix_all_mix_loss':
        from models.enhancement.model_E_UNet_mpvit_mix import TwoBranchNetwork_SA
        model_E = TwoBranchNetwork_SA()
        model_E = model_E.to(device)
        args.pretrained = '/home/catchacat/workspace/Underwater_enhancement/my_TFUIE_new/checkpoints/my_UIE_mpvit_sa_mix_vgg/my_UIE_mpvit_sa_mix-my_UIE-SUIM-lr=0.0001-wd=0.001-bs=4-01-07-2026-10-43-28-best_ssim.pth'

    elif args.model_E == 'my_UIE_mpvit_sa_mix_16_mix_loss':
        from models.enhancement.model_E_UNet_mpvit_mix import TwoBranchNetwork_SA
        model_E = TwoBranchNetwork_SA()
        model_E = model_E.to(device)
        args.pretrained = '/home/catchacat/workspace/Underwater_enhancement/my_TFUIE_new/checkpoints/my_UIE_mpvit_sa_mix_vgg/my_UIE_mpvit_sa_mix-my_UIE-SUIM-lr=0.0001-wd=0.001-bs=4-01-08-2026-21-37-21-best_ssim.pth'
    elif args.model_E == 'my_UIE_mpvit_sa_mix_no_train_loss':
        from models.enhancement.model_E_UNet_mpvit_mix import TwoBranchNetwork_SA
        model_E = TwoBranchNetwork_SA()
        model_E = model_E.to(device)
        args.pretrained = '/home/catchacat/workspace/Underwater_enhancement/my_TFUIE_new/checkpoints/my_UIE_mpvit_sa_mix_vgg/my_UIE_mpvit_sa_mix-my_UIE-SUIM-lr=0.0001-wd=0.001-bs=4-01-12-2026-20-03-33-best_ssim.pth'



    from models.segmentation.vgg_unet import VGG16Unet, VGG13Unet
    from models.segmentation.resnet_unet import ResNet50Unet

    model_S = VGG16Unet(n_channels=3, n_classes=8, pretrained=True)
    #model_S = ResNet50Unet(n_channels=3, n_classes=8, pretrained=True)
    model_S = model_S.to(device)



    from dataloader_seg import Tested_Set
    # dataset_dir = '/home/catchacat/data/UnderwaterDatasets/UnderwaterDetection/SUIM/SUIM_fix/train_val/images/'
    # dataset_dir = '/home/catchacat/data/UnderwaterDatasets/UnderwaterDetection/Seg_UIIS/UIIS/UDW/train/'
    dataset_dir = '/home/catchacat/data/UnderwaterDatasets/UnderwaterEnhancement/UIEB/valid/images'
    args.classes = 8
    dataset = Tested_Set(file_path = dataset_dir, status='valid', augmentation=False, angle=0, size_h=args.crop_size, size_w=args.crop_size, hflip_p=0)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, drop_last=True, batch_size=1, num_workers=8, pin_memory=True)

    # logging.info(f'model params: %d' % count_parameters(model))
    # print('there are total %s batches for train' % (len(train_loader)))
    # print('there are total %s batches for val' % (len(val_loader)))
    # # >>> [pretrained] reload pretrained models
    if args.pretrained:
        pre_trained_state_dict = torch.load(args.pretrained)
        # for key in pre_trained_state_dict['model_E'].keys():
        #     print(key)
     
        model_E.load_state_dict(pre_trained_state_dict['model_E'])
        model_S.load_state_dict(pre_trained_state_dict['model_S'])
        #start_epoch = ckpt['epoch'] + 1

    model_E.to(device=device), model_S.to(device=device)

    # >>> print model
    # image = torch.randn(1,3,384,384).cuda()
    # writer.add_graph(model, (image,))

    model_E.eval(), model_S.eval()
    num_val_batches = len(dataloader)

    epoch_valid_psnr = []
        
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # times = []

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
                # starter.record()
                outputs = model_E(images, feat_raw)
                # ender.record()
            else:
                outputs = model_E(images)
            # temp_uciqe, N = compute_uciqe(outputs)
            # valid_average_uciqe.update(temp_uciqe, N)
            # temp_uiqm, N = compute_uiqm(outputs)
            # valid_average_uiqm.update(temp_uiqm, N)

            # torch.cuda.synchronize()
            # elapsed_time_ms = starter.elapsed_time(ender)
            # times.append(elapsed_time_ms)

            if args.draw_images:
                for i, (output, image_name) in enumerate(zip(outputs, images_names)):
                    output = output.unsqueeze(0)
                    output = F.interpolate(output, size=(original_h, original_w), mode='bicubic', align_corners=False)
                    output = output.squeeze(0)

                    output = output.permute(1, 2, 0).cpu().numpy()
                    output = (output * 255).astype(np.uint8)
                    output = Image.fromarray(output)
                    
                    # image.save(f'{args.save_dir}/{args.model}/{os.path.basename(image_name)}_src.png')
                    # output.save(f'/home/catchacat/data/UnderwaterDatasets/UnderwaterDetection/SUIM/Enhance/my_TFUIE_notrain_loss/train_val/{os.path.basename(image_name)}')
                    # output.save(f'/home/catchacat/data/UnderwaterDatasets/UnderwaterDetection/Seg_UIIS/Enhance/Enhance_FullSize/train/{os.path.basename(image_name)}')
                    output.save(f'/home/catchacat/workspace/Underwater_enhancement/my_TFUIE_new/outputs/UIEB_my_UIE_mpvit_sa_mix/{os.path.basename(image_name)}')


                    
                    # output.save(f'{args.save_dir}/{args.model}/{os.path.basename(image_name)}_output.png')
                    # image_gt.save(f'{args.save_dir}/{args.model}/{os.path.basename(image_name)}_gt.png')

    # print(f"Average inference time: {np.mean(times):.3f} ms")
    

