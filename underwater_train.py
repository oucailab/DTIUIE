import argparse

import os
import numpy as np
import random
import datetime

import logging
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler

import torch.nn.functional as F

from dataloader_seg import Paired_Set
from uie_eval import compute_psnr_ssim, AverageMeter,  to_psnr, compute_uciqe, compute_uiqm
from seg_eval import SegmentationMetric
from utils.logger import MetricLogger, SmoothedValue
from utils.lr_scheduler import CosineAnnealingRestartLR
try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


torch.cuda.empty_cache()
current_time = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

from loss import FeatureLoss, L1Loss, AUXCELoss, MSELoss

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
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument("--seed", type=int, default=2022)

    ### >>> [data] set data info.
    parser.add_argument('--database', default='TIUIED', type=str, help='database name')
    parser.add_argument('--crop_size', type=int, default=256, help='input image size for training (default: 256)')

    ### >>> [model] set model info.
    parser.add_argument('--model_E', default='model_dtiuie', type=str, help='model name')
    parser.add_argument('--model_S', default='vgg16', type=str, help='model name')

    parser.add_argument('--resume', default='False', type=str, help='if resume')
    parser.add_argument('--resume_path', default='/path/to/your/net.pth', type=str, help='if resume')
    parser.add_argument('--use_pretain', default='False', type=str, help='use pretained model')
    parser.add_argument('--pretrained', default=None, type=str, help='path to latest checkpoint (default: None)')

    ### >>> [val] set validation criterion info.
    parser.add_argument('--val_criterion', default='mIoU', type=str)

    ### >>> [train] set training info.
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 3000)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('-g', '--gpus', type=int, default=1, metavar='N')

    ### >>> [lr] set learning rate decay info.
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay (default: 0.0)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--decay_interval', type=list, default=[150,180], help='learning rate decay interval (default: 100)')
    parser.add_argument('--decay_ratio', type=float, default=0.8, help='learning rate decay ratio (default: 0.8)')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='number of warmup epochs (default: 0)')

    ### >>> [save] set log and model save path info.
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs", help="log directory for Tensorboard log output")
    parser.add_argument('--save_best_checkpoint', default=True, type=bool, help='Save the best model checkpoints during training.')
    parser.add_argument('--save_iter_checkpoint', default=False, type=bool, help='Save the iteration model checkpoints during training.')
    parser.add_argument('--save_final_checkpoint', default=True, type=bool, help='Save the final model checkpoints during training.')


    ### >>> [loss] Enhancement and Segmentation loss function.
    parser.add_argument("--loss_enc_pixel", type=str, default="L1Loss")
    parser.add_argument("--loss_enc_pixel_weight", type=float, default=1.0)
    parser.add_argument("--loss_enc_feat", type=str, default="FeatureLoss")
    parser.add_argument("--loss_enc_feat_weight", type=float, default=1.0)
    parser.add_argument("--loss_seg_raw", type=str, default="AUXCELoss")
    parser.add_argument("--loss_seg_raw_weight", type=float, default=0.34)
    parser.add_argument("--loss_seg_enc", type=str, default="AUXCELoss")
    parser.add_argument("--loss_seg_enc_weight", type=float, default=0.34)
    parser.add_argument("--loss_seg_mix", type=str, default="AUXCELoss")
    parser.add_argument("--loss_seg_mix_weight", type=float, default=0.34)
    parser.add_argument("--loss_seg_sa", type=str, default="AUXCELoss")
    parser.add_argument("--loss_seg_sa_weight", type=float, default=1.0)


    parser.add_argument('--info', default=None, type=str, help='additional info')
    return parser.parse_args()

def train_model(args):
    device, available_gpus = get_available_devices(args.gpus)
    args.local_rank = -1


    # >>> [checkpoints] set checkpoints and results dir.
    # >>> set checkpoints and results dir.
    args.log_format_str = '{}/{}-{}-{}-lr={}-wd={}-bs={}-{}'.format(args.log_dir, args.model_E, args.model_S, args.database, args.learning_rate, args.weight_decay, args.batch_size, current_time)
    if args.save_best_checkpoint:
        os.makedirs('checkpoints/{}_vgg'.format(args.model_E), exist_ok=True)
        args.best_model_file = 'checkpoints/{}_vgg/{}-{}-{}-lr={}-wd={}-bs={}-{}-best'.format(args.model_E, args.model_E, args.model_S, args.database, args.learning_rate, args.weight_decay, args.batch_size, current_time)
    if args.save_final_checkpoint or args.save_iter_checkpoint:
        os.makedirs('results/{}_vgg'.format(args.model_E), exist_ok=True)
        args.final_model_file = 'results/{}_vgg/{}-{}-{}-lr={}-wd={}-bs={}-{}-final'.format(args.model_E, args.model_E, args.model_S, args.database, args.learning_rate, args.weight_decay, args.batch_size, current_time)
        args.save_result_file = 'results/{}_vgg/{}-{}-{}-lr={}-wd={}-bs={}-{}'.format(args.model_E, args.model_E, args.model_S, args.database, args.learning_rate, args.weight_decay, args.batch_size, current_time)

    # >>> [random] set random seeds info.
    # torch.manual_seed(args.seed)  
    # torch.cuda.manual_seed_all(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    # >>> [LOG] set logging info.
    args.tensorboard_dir = 'tensorboardLogs' 
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    writer = SummaryWriter(log_dir='{}'.format(args.log_format_str))

    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.learning_rate}
        Weight decay:    {args.weight_decay}
        Best Checkpoints:     {args.save_best_checkpoint}
        Liter Checkpoints:    {args.save_iter_checkpoint}
        Final Checkpoints:    {args.save_final_checkpoint}
        Device:          {device.type}
        Mixed Precision: {args.amp}
    ''')


    ### >>> [DATA] Set up data loader.
    if args.database == 'TIUIED':
        dir_train = "./dataset/trainval/"
        dir_valid = "./dataset/test/"
        args.classes = 8
    train_dataset = Paired_Set(file_path = dir_train, status='train', augmentation=True, angle=2, size_h=args.crop_size, size_w=args.crop_size, hflip_p=0.5)
    valid_dataset = Paired_Set(file_path = dir_valid, status='valid', augmentation=False, angle=0, size_h=args.crop_size, size_w=args.crop_size, hflip_p=0)
    valid_dataset.mask_values = train_dataset.mask_values
    

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, batch_size=1, num_workers=8, pin_memory=True)
    print("len train loader: ", len(train_loader), "len val loader", len(valid_loader))

    # >>> [MODEL] set model info
    if  args.model_E == 'model_dtiuie':
        from models.enhancement.model_dtiuie import DTIUIE
        model_E = DTIUIE()
        model_E = model_E.to(device)

        
    from models.segmentation.vgg_unet import VGG16Unet   
    model_S = VGG16Unet(n_channels=3, n_classes=8, pretrained=True)
    model_S = model_S.to(device)

    model_SA = VGG16Unet(n_channels=3, n_classes=8, pretrained=True)
    model_SA = model_SA.to(device)


    writer.add_text("training/enhancement_model", str(args.model_E))
    writer.add_text("training/segmentation_model", str(args.model_S))
    writer.add_text("training/enhancement_model_details", str(model_E))

    # model = torch.nn.DataParallel(model, device_ids=device_ids)

    ### >>> [LOSS] Set up loss function.
    if args.loss_enc_feat == 'FeatureLoss':
        loss_func_feat = FeatureLoss(layer_weights={'C5': 1.0}, loss_weight=args.loss_enc_feat_weight, criterion='l1')
    else: loss_func_feat = None

    if args.loss_enc_pixel == 'L1Loss':
        loss_func_pixel = L1Loss(loss_weight=args.loss_enc_pixel_weight)
    elif args.loss_enc_pixel == 'MSELoss':
        loss_func_pixel = MSELoss(loss_weight=args.loss_enc_pixel_weight)
    else: loss_func_pixel = None

    if args.loss_seg_raw == 'AUXCELoss':
        loss_func_seg_raw = nn.CrossEntropyLoss() #AUXCELoss(loss_weight=args.loss_seg_raw_weight)
    else: loss_func_seg_raw = None

    if args.loss_seg_sa == 'AUXCELoss':
        loss_func_seg_sa = nn.CrossEntropyLoss() #AUXCELoss(loss_weight=args.loss_seg_raw_weight)
        print("loss_func_seg_sa: ", loss_func_seg_sa)
    else: loss_func_seg_sa = None

    if args.loss_seg_enc == 'AUXCELoss':
        loss_func_seg_enc = nn.CrossEntropyLoss() #AUXCELoss(loss_weight=args.loss_seg_enc_weight)
    else: loss_func_seg_enc = None

    if args.loss_seg_mix == 'AUXCELoss':
        loss_func_seg_mix = nn.CrossEntropyLoss() #AUXCELoss(loss_weight=args.loss_seg_mix_weight)
    else: loss_func_seg_mix = None



    ### >>> [OPTIM] Set up the optimizer, the learning rate scheduler and the loss scaling for AMP
    # Optimizers & LR schedulers
    optimizer = torch.optim.AdamW(model_E.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999))
    optimizer_seg = torch.optim.SGD(model_S.parameters(), lr=2e-2, momentum=0.9, weight_decay=1e-4)
    optimizer_sa = torch.optim.SGD(model_SA.parameters(), lr=2e-2, momentum=0.9, weight_decay=1e-4)

    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay_interval, gamma=args.decay_ratio)    # after every "decay_interval", lr = lr * decay_ratio
    #scheduler_seg = optim.lr_scheduler.MultiStepLR(optimizer_seg, milestones=args.decay_interval, gamma=args.decay_ratio)    # after every "decay_interval", lr = lr * decay_ratio
    scheduler = CosineAnnealingRestartLR(optimizer, periods=[10, 90], restart_weights=[1, 1], eta_min=1e-6)
    scheduler_seg = CosineAnnealingRestartLR(optimizer_seg, periods=[10, 90], restart_weights=[1, 1], eta_min=1e-4)
    scheduler_sa = CosineAnnealingRestartLR(optimizer_seg, periods=[10, 90], restart_weights=[1, 1], eta_min=1e-4)


    scaler = GradScaler()
    scaler_seg = GradScaler()
    scaler_sa = GradScaler()


    ### >>> [VAL] Set up best validation criterion.
    metric_name = ['psnr', 'ssim', 'miou', 'dice', 'acc', 'uciqe', 'uiqm']
    metrics = {name: float('-inf') for name in metric_name}
    save_metric_name = ['ssim', 'miou']
    best_criterions = {save_name: {"score": float('-inf'), 
                                   "epoch": -1, 
                                   "metrics": {name: float('-inf') for name in metric_name}} 
                                            for save_name in save_metric_name}
    print("best_criterions: ", best_criterions)
    print(best_criterions['ssim']["metrics"]['psnr'])
    best_checkpoints = {name: None for name in metrics.keys()}


    best_val_criterion, best_epoch = -1, -1  # larger, better, e.g., SROCC or PLCC. If RMSE is used, best_val_criterion <- 10000 
    best_criterion = {}
    global_step = 0
    global_epoch_curiter = 0


    ### >>>>>>>>>> 5. Begin training >>>>>>>>>>>>>>>
    ## >>> 5.1. Begin training in epochs.
    for epoch in range(1, args.epochs + 1):
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr_sr", SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("lr_seg", SmoothedValue(window_size=1, fmt="{value}"))

        

        epoch_losses_meter = AverageMeter()
        epoch_losses_E_meter = AverageMeter()
        epoch_losses_S_meter = AverageMeter()


        train_average_psnr_meter = AverageMeter()
        train_average_ssim_meter = AverageMeter()

        ## >>> Set model to training mode.
        model_S.train(), model_E.train(), model_SA.train()

        torch.cuda.empty_cache()     
        ## >>> 5.2. Begin training in iterations <<<<< .
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{args.epochs}', unit='batch', ncols=150) as pbar:
            ## >>> Rezip unsupervised_loader and supervised_loader to train_loader. 
            for i, batch in enumerate(train_loader):
                images = batch['image'].to(device)
                images_gt = batch['image_gt'].to(device)
                images_mask = batch['image_mask'].to(device) 
                

                # [Phase 0]: train SA Segmentation model.
                if 'sa' in args.model_E:
                    optimizer_sa.zero_grad()
                    loss_sa_total = 0
                    pred_mask_raw = model_SA(images, return_feats=False)

                    if loss_func_seg_sa is not None:
                        loss_seg_sa = loss_func_seg_sa(pred_mask_raw, images_mask)
                        writer.add_scalar("training/iteration_loss_seg_sa", loss_seg_sa.item(), global_step)
                        loss_sa_total += loss_seg_sa

                    scaler_sa.scale(loss_sa_total).backward()
                    scaler_sa.step(optimizer_sa)
                    scaler_sa.update()


                # [Phase 1]: train Enhancement model.
                # for p in model_S.parameters(): p.requires_grad = False
                if 'sa' in args.model_E:
                    model_SA.eval()
                    _, feat_raw = model_SA(images, return_feats=True)
                    images_enhance = model_E(images, feat_raw)
                    model_SA.train()
                else:
                    images_enhance = model_E(images)

                
                optimizer.zero_grad()
                loss_enc_total = 0

                if loss_func_pixel is not None:
                    loss_enc_pixel = loss_func_pixel(images_enhance, images_gt)
                    # metric_logger.meters["loss_enc_pixel"].update(loss_enc_pixel.item())
                    writer.add_scalar("training/iteration_loss_enc_pixel", loss_enc_pixel.item(), global_step)
                    loss_enc_total += loss_enc_pixel

                if epoch > args.warmup_epochs:
                    if loss_func_feat is not None:
                        model_S.eval()
                        _, feat_enc = model_S(images_enhance, return_feats=True)
                        _, feat_gt = model_S(images_gt, return_feats=True)
                        model_S.train()

                        loss_enc_feature = loss_func_feat(feat_enc, feat_gt)
                        loss_enc_total += loss_enc_feature
                        # metric_logger.meters["loss_enc_feature"].update(loss_enc_feature.item())
                        writer.add_scalar("training/iteration_loss_enc_feature", loss_enc_feature.item(), global_step)
                
                scaler.scale(loss_enc_total).backward()
                scaler.step(optimizer)
                scaler.update()


                # [Phase 2]: train Segmentation model.
                if 'sa' in args.model_E:
                    images_enhance = model_E(images, feat_raw).detach()
                else:
                    images_enhance = model_E(images).detach()

                for p in model_S.parameters(): p.requires_grad = True
                optimizer_seg.zero_grad()
                loss_seg_total = 0

                # >> loss for segmentation model of enhancement results.
                if loss_func_seg_enc is not None:
                    pred_mask_enc = model_S(images_enhance)
                    loss_seg_enc = loss_func_seg_enc(pred_mask_enc, images_mask)
                    loss_seg_total += loss_seg_enc

                    # metric_logger.meters["loss_seg_enc"].update(loss_seg_enc.item())
                    writer.add_scalar("training/iteration_loss_seg_enc", loss_seg_enc.item(), global_step)

                # >> loss for segmentation model of original images.
                if loss_func_seg_raw is not None:
                    pred_mask_gt = model_S(images_gt)
                    loss_seg_ori = loss_func_seg_raw(pred_mask_gt, images_mask)
                    loss_seg_total += loss_seg_ori

                    # metric_logger.meters["loss_seg_raw"].update(loss_seg_ori.item())
                    writer.add_scalar("training/iteration_loss_seg_raw", loss_seg_ori.item(), global_step)

                # >> loss for segmentation model of mixed images.
                if loss_func_seg_mix is not None:
                    batch_size = images_enhance.shape[0]
                    mix_mask_scale_factor = images_enhance.shape[-1] // 8
                    #mix_mask = F.interpolate((torch.randn(batch_size,1,8,8)).bernoulli_(p=0.5), scale_factor=mix_mask_scale_factor, mode='nearest').to(device)
                    mix_mask = F.interpolate(torch.sigmoid(torch.randn(batch_size,1,8,8)), scale_factor=mix_mask_scale_factor, mode='nearest').to(device)

                    images_mix = images_enhance * mix_mask + images_gt * (1 - mix_mask)
                    pred_mask_mix = model_S(images_mix)
                    loss_seg_mix = loss_func_seg_mix(pred_mask_mix, images_mask)
                    loss_seg_total += loss_seg_mix

                    # metric_logger.meters["loss_seg_mix"].update(loss_seg_mix.item())
                    writer.add_scalar("training/iteration_loss_seg_mix", loss_seg_mix.item(), global_step)

                scaler_seg.scale(loss_seg_total).backward()
                scaler_seg.step(optimizer_seg)
                scaler_seg.update()

                ## >>> [4] update loss meter.
                # epoch_losses_meter.update(loss.item())
                epoch_losses_E_meter.update(loss_enc_total.item())
                epoch_losses_S_meter.update(loss_seg_total.item())


                pbar.update(1)
                # pbar.set_postfix(**{'loss_E': loss_E.item(), 'loss_S': (loss_S.item()), 'ls_edge':(loss_function_edge.item()), 'lr': optimizer.param_groups[0]['lr']})
                pbar.set_postfix(**{'loss_E': loss_enc_total.item(), 'loss_S': (loss_seg_total.item()), 'lr': optimizer.param_groups[0]['lr']})

                writer.add_scalar("training/iteration_loss_E", loss_enc_total.item(), global_step)
                writer.add_scalar("training/iteration_loss_S", loss_seg_total.item(), global_step)
                global_step = global_step + 1

                ## >>> [5] finish iteration training. <<<<<<

        scheduler.step()
        scheduler_seg.step()
        writer.add_scalar("training/epoch_loss_E", epoch_losses_E_meter.avg, epoch)
        writer.add_scalar("training/epoch_loss_S", epoch_losses_S_meter.avg, epoch)
        writer.add_scalar("training/epoch_loss", epoch_losses_meter.avg, epoch)
        writer.add_scalar("training/learning_rate", optimizer.param_groups[0]['lr'], epoch)
        
        ## >> finish epoch training.
        ## >> 5.3. Evaluate the model on the validation set.
        metrics['psnr'], metrics['ssim'],  metrics['uciqe'], metrics['uiqm'], \
            metrics['acc'], metrics['miou'], metrics['dice'] = evaluate_model(args, model_E, model_SA, valid_loader, device, args.model_E)
        

        logging.info('Train PSNR: {:.4f}, Validation PSNR: {:.4f}, SSIM: {:.4f}, UCIQE: {:.4f}, UIQM: {:.4f}'.format(train_average_psnr_meter.avg, metrics['psnr'], metrics['ssim'], metrics['uciqe'], metrics['uiqm']))
        logging.info('Validation pix_acc: {}, dice_score: {}, miou: {}'.format(metrics['acc'], metrics['dice'], metrics['miou']))
        logging.info('Epoch: {}, T_Loss_E: {:.4f}, T_Loss_S: {:.4f}, T_Loss: {:.4f},' 
                     .format(epoch, epoch_losses_E_meter.avg, epoch_losses_S_meter.avg, epoch_losses_meter.avg))

        for name in metrics.keys():
            writer.add_scalar("validation/epoch_{}".format(name), metrics[name], epoch)


        # >>> 5.4. save iteration model.
        if args.save_iter_checkpoint:
            # save checkpoint every 10 epoch.
            if epoch % 10 == 0:
                checkpoint = {  'model_E': model_E.state_dict(),                'model_S': model_S.state_dict(),        'model_SA': model_SA.state_dict(),
                                'optimizer_G': optimizer.state_dict(),          'scheduler_G' : scheduler.state_dict(),
                                'scaler' : scaler.state_dict(),
                                'epoch': epoch
                            }
                checkpoint_model_file = 'results/{}/{}-{}-lr={}-wd={}-bs={}-epoch={}.pth'.format(current_time, args.model, args.database, args.learning_rate, args.weight_decay, args.batch_size, epoch)
                torch.save(checkpoint, checkpoint_model_file)
                logging.info('--- Save model criterion: @epoch: {}'.format(epoch))

        for save_name in save_metric_name:
            if metrics[save_name] > best_criterions[save_name]["score"]:
                best_criterions[save_name]["score"] = metrics[save_name]
                best_criterions[save_name]["epoch"] = epoch
                best_criterions[save_name]["metrics"] = {name: metrics[name] for name in metric_name}
                
                if args.save_best_checkpoint == True:
                    checkpoint = {  'model_E': model_E.state_dict(),                'model_S': model_S.state_dict(),    'model_SA': model_SA.state_dict(), 
                                    'optimizer_G': optimizer.state_dict(),          'scheduler_G' : scheduler.state_dict(),
                                    'scaler' : scaler.state_dict(),
                                    'epoch': epoch
                                }
                    torch.save(checkpoint, f"{args.best_model_file}_{save_name}.pth")
                    logging.info('--- Save current best model @{} ({}): {:.5f} @epoch: {}'.format(save_name, save_name, best_criterions[save_name]["score"], best_criterions[save_name]["epoch"]))
            else:
                logging.info('Model is not updated @val_criterion ({}): {:.5f} @epoch: {}'.format(save_name, metrics[save_name], epoch))

    # TRAINING_COMPLETED.
    final_psnr, final_ssim, final_uciqe, final_uiqm, \
        final_pix_acc, final_mean_iou, final_dice_score, \
                = evaluate_model(args, model_E, model_S, valid_loader, device, args.model_E)   

    logging.info('*Final* Validation PSNR: {:.4f}, SSIM: {:.4f}, UCIQE: {:.4f}, UIQM: {:.4f}'.format(final_psnr, final_ssim, final_uciqe, final_uiqm))
    logging.info('*Final* Validation pix_acc: {}, dice_score: {}, miou: {}'.format(final_pix_acc, final_dice_score, final_mean_iou))

    for save_name in save_metric_name:
        logging.info('*Best_{}* Validataion PSNR: {:.4f}, SSIM: {:.4f}, UCIQE: {:.4f}, UIQM: {:.4f} at epoch: {}'
                        .format(save_name, best_criterions[save_name]["metrics"]['psnr'], best_criterions[save_name]["metrics"]['ssim'], 
                                           best_criterions[save_name]["metrics"]['uciqe'], best_criterions[save_name]["metrics"]['uiqm'], best_criterions[save_name]['epoch']))
        logging.info('*Best_{}* Validation dice_score: {}, miou: {}, pixACC: {}'
                        .format(save_name, best_criterions[save_name]["metrics"]['dice'],  best_criterions[save_name]["metrics"]['miou'],  best_criterions[save_name]["metrics"]['acc']))

    logging.info('at dict:{}'.format(current_time))

    if args.save_final_checkpoint:
        checkpoint = {  'model_E': model_E.state_dict(),                'model_S': model_S.state_dict(),  'model_SA': model_SA.state_dict(),
                                'optimizer_G': optimizer.state_dict(),          'scheduler_G' : scheduler.state_dict(),
                                'scaler' : scaler.state_dict(),
                                'epoch': epoch
                            }
        torch.save(checkpoint, args.final_model_file)
        logging.info('Save final model criterion @epoch: {}'.format(epoch))

    writer.add_text("validation/best_enhance", '*Best_{}* Validataion PSNR: {:.4f}, SSIM: {:.4f}, UCIQE: {:.4f}, UIQM: {:.4f} at epoch: {}'
                        .format(save_name, best_criterions[save_name]["metrics"]['psnr'], best_criterions[save_name]["metrics"]['ssim'], 
                                           best_criterions[save_name]["metrics"]['uciqe'], best_criterions[save_name]["metrics"]['uiqm'], best_criterions[save_name]['epoch']))
    writer.add_text("validation/best_segmentation", '*Best_{}* Validataion PSNR: {:.4f}, SSIM: {:.4f}, UCIQE: {:.4f}, UIQM: {:.4f} at epoch: {}'
                        .format(save_name, best_criterions[save_name]["metrics"]['psnr'], best_criterions[save_name]["metrics"]['ssim'], 
                                           best_criterions[save_name]["metrics"]['uciqe'], best_criterions[save_name]["metrics"]['uiqm'], best_criterions[save_name]['epoch']))
    writer.close()

def evaluate_model(args, model_E, model_SA, dataloader, device, name):
    model_E.eval(), model_SA.eval()
    num_val_batches = len(dataloader)

    valid_average_psnr = AverageMeter()
    valid_average_ssim = AverageMeter()
    valid_average_uciqe = AverageMeter()
    valid_average_uiqm = AverageMeter()

    valid_seg_metric = SegmentationMetric(8, softmax=True)
        
    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', ncols=150, leave=True):
            images = batch['image'].to(device)
            images_gt = batch['image_gt'].to(device)
            images_mask = batch['image_mask'].to(device)    # label: semantic label

            # Generate output
            if 'sa' in name:
                _, feat_raw = model_SA(images, return_feats=True)
                images_enhance = model_E(images, feat_raw)
            else:
                images_enhance = model_E(images)
            
            images_pred_mask = model_SA(images_enhance)

            # update evaluation losses

            temp_psnr, temp_ssim, N = compute_psnr_ssim(images_enhance, images_gt)
            valid_average_psnr.update(temp_psnr, N)
            valid_average_ssim.update(temp_ssim, N)
            valid_seg_metric.update(images_pred_mask, images_mask)

        torch.cuda.empty_cache()

    pix_acc, mean_iou, dice_score = valid_seg_metric.get()
    return  valid_average_psnr.avg, valid_average_ssim.avg, valid_average_uciqe.avg, valid_average_uiqm.avg, \
                pix_acc, mean_iou, dice_score


if __name__ == '__main__':
    args = get_args()
    args.local_rank = -1

    train_model(args=args)




