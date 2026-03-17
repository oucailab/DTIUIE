'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-10-30 13:46:57
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-03-15 12:32:40
FilePath: /ouc/workspace/Single_image_regression/my_NR-IQM/loss.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
from torchvision import transforms
from torchvision import models


_reduction_modes = ['none', 'mean', 'sum']


def l1_loss(pred, target):
    return F.l1_loss(pred, target)


def mse_loss(pred, target):
    return F.mse_loss(pred, target)


def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """

        return self.loss_weight * l1_loss(pred, target)
    

class MSELoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """

        return self.loss_weight * mse_loss(pred, target)
    
    
class FeatureLoss(nn.Module):
    """Feature Loss
    """

    def __init__(self,
                 layer_weights,
                 loss_weight=1.0,
                 criterion='l1'):

        super().__init__()
        self.loss_weight = loss_weight
        self.layer_weights = layer_weights

        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, feats_sr, feats_hr):
        """Forward function.

        Args:
            x_features List(Tensor): TODO
            gt_features List(Tensor): TODO

        Returns:
            Tensor: Forward results.
        """
        # calculate perceptual loss
        if self.loss_weight > 0:
            feature_loss = 0
            for k in self.layer_weights.keys():
                if isinstance(feats_hr[k], list):
                    for idx in range(len(feats_sr[k])):
                        feature_loss += self.criterion(feats_sr[k][idx], feats_hr[k][idx]) * self.layer_weights[k]
                elif isinstance(feats_hr[k], dict):
                    for kk in feats_hr[k].keys():
                        feature_loss += self.criterion(feats_sr[k][kk], feats_hr[k][kk]) * self.layer_weights[k]
                else:
                    feature_loss += self.criterion(feats_sr[k], feats_hr[k]) * self.layer_weights[k]
            feature_loss *= self.loss_weight
        else:
            feature_loss = None

        return feature_loss


class AUXCELoss(nn.Module):
    """Cross Entropy loss with Auxilary loss.

    Args:
        loss_weight (float): Loss weight for CE loss. Default: 1.0.
        loss_weight (float): Loss weight for auxilrary loss. Default: 0.5.
    """
    def __init__(self, loss_weight=1.0, aux_loss_weight=0.5, ignore_index=-100):
        super().__init__()
        self.loss_weight = loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.ignore_index = ignore_index
    
    def forward(self, pred, target, **kwargs):
        losses = {}
        for name, x in pred.items():
            losses[name] = F.cross_entropy(x, target, ignore_index=self.ignore_index)

        if len(losses) == 1:
            return self.loss_weight * losses["out"]

        return self.loss_weight * (losses["out"] + self.aux_loss_weight * losses["aux"])
    
class SobelEdgeLoss(nn.Module):
    '''
    Sobel Edge Loss function.
    Reference: 'Attention Network for Non-Uniform Deblurring'
    '''
    def __init__(self, device):
        super(SobelEdgeLoss, self).__init__()
        sobel_h = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_v = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_h = sobel_h.to(device)
        self.sobel_v = sobel_v.to(device)

    def forward(self, pred, target):
        """
        Compute the edge loss between the ground truth `s` and the generated image `G(b)`.

        Arguments:
        s -- Ground truth image tensor of shape (B, C, H, W)
        G_b -- Generated image tensor of shape (B, C, H, W)

        Returns:
        loss -- Calculated edge loss
        """
        loss_h = 0
        loss_v = 0
        for i in range(pred.size(1)):  # Iterate over each channel
            # Apply Sobel filters to get horizontal and vertical gradients
            grad_target_h = F.conv2d(target[:, i:i+1, :, :], self.sobel_h, padding=1)
            grad_target_v = F.conv2d(target[:, i:i+1, :, :], self.sobel_v, padding=1)
            grad_pred_h = F.conv2d(pred[:, i:i+1, :, :], self.sobel_h, padding=1)
            grad_pred_v = F.conv2d(pred[:, i:i+1, :, :], self.sobel_v, padding=1)

            # Calculate L1 loss for horizontal and vertical gradients
            loss_h += torch.abs(grad_target_h - grad_pred_h)
            loss_v += torch.abs(grad_target_v - grad_pred_v)

        # Normalize by height (H) and width (W)
        H, W = target.shape[2], target.shape[3]
        loss = (loss_h.sum() + loss_v.sum()) / (H * W)
        loss = loss / pred.size(1)  # Normalize by number of channels
        
        return loss



