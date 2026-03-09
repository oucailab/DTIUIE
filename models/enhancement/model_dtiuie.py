import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np

import torchvision.transforms.functional as TF

import numbers
from einops import rearrange
from torch import nn, einsum

def normalize_img(img):
    if torch.max(img) > 1 or torch.min(img) < 0:
        # img: b x c x h x w
        b, c, h, w = img.shape
        temp_img = img.view(b, c, h*w)
        im_max = torch.max(temp_img, dim=2)[0].view(b, c, 1)
        im_min = torch.min(temp_img, dim=2)[0].view(b, c, 1)

        temp_img = (temp_img - im_min) / (im_max - im_min + 1e-7)
        
        img = temp_img.view(b, c, h, w)
    
    return img



class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,  bias=True):
        super(Conv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x
    

class ResidualConvLayer(nn.Module):
    def __init__(self, in_dim, dim, mid_dim=None, kernel_size=3, gate_act=nn.Sigmoid):
        super().__init__()
        self.dim = dim
        if mid_dim is None:
            mid_dim = dim

        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_dim),
            nn.Conv2d(mid_dim, dim, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        out = self.convs(x)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_dim, dim, mid_dim=None, kernel_size=3, gate_act=nn.Sigmoid):
        super().__init__()
        self.dim = dim
        if mid_dim is None:
            mid_dim = dim

        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, mid_dim, kernel_size=kernel_size, padding=padding, bias=False),
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, dim, kernel_size=kernel_size, padding=padding, bias=False),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


"""
Conv-Transformer Block.
Reference: https://github.com/RQ-Wu/UnderwaterRanker/blob/master/all_model/URanker/model.py
"""

class Conv2d_BN(nn.Module):
    """Convolution with BN module."""
    """ Reference https://github.com/youngwanLEE/MPViT/blob/main/mpvit.py#L82"""

    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bn_weight_init=1, norm_layer=nn.BatchNorm2d, act_layer=None,
                ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_ch)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        """foward function"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x

class TransPatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, in_dim, embed_dim, patch_size=3, stride=1, isPool=False, act_layer=nn.Hardswish):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.pe = Conv2d_BN(in_dim, embed_dim, kernel_size=patch_size, padding=(patch_size[0]-1)//2, stride=2 if isPool else 1)

    def forward(self, x):
        out = self.pe(x)
        return out

class Mlp(nn.Module):
    """ Feed-forward network (FFN, a.k.a. MLP) class. """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class GatedFeedforward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.gate = nn.Linear(in_features, hidden_features)

    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        x = self.fc1(x)
        x = self.act(x)
        x = x * gate
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvRelPosEnc(nn.Module):
    """Convolutional relative position encoding."""
    def __init__(self, Ch, h, window):
        """Initialization.
            Ch: Channels per head.
            h: Number of heads.
            window: Window size(s) in convolutional relative positional encoding.
                    It can have two forms:
                    1. An integer of window size, which assigns all attention headswith the same window size in ConvRelPosEnc.
                    2. A dict mapping window size to #attention head splits (e.g. {window size 1: #attention head split 1, window size
                                        2: #attention head split 2})
                    It will apply different window size to the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []

        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, q, v, size):
        """foward function"""
        B, h, N, Ch = q.shape
        H, W = size

        # We don't use CLS_TOKEN
        q_img = q
        v_img = v

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img = rearrange(v_img, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W).contiguous()
        # Split according to channels.
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        conv_v_img = rearrange(conv_v_img, "B (h Ch) H W -> B h (H W) Ch", h=h).contiguous()

        EV_hat_img = q_img * conv_v_img
        EV_hat = EV_hat_img
        return EV_hat

class FactorAtt_ConvRelPosEnc(nn.Module):
    """ Factorized attention with convolutional relative position encoding class. """
    """ Reference https://github.com/mlpc-ucsd/CoaT/blob/main/src/models/coat.py#L119 """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., shared_crpe=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)                                           # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
        q, k, v = qkv[0], qkv[1], qkv[2]                                                 # Shape: [B, h, N, Ch].



        # Factorized attention.
        k_softmax = k.softmax(dim=2)                                                     # Softmax on dim N.
        k_softmax_T_dot_v = einsum('b h n k, b h n v -> b h k v', k_softmax, v)          # Shape: [B, h, Ch, Ch].
        factor_att        = einsum('b h n k, b h k v -> b h n v', q, k_softmax_T_dot_v)  # Shape: [B, h, N, Ch].

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)                                                # Shape: [B, h, N, Ch].

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)                                           # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x   # Shape: [B, N, C].


class FactorAtt_Mix_ConvRelPosEnc(nn.Module):
    """ Factorized attention with convolutional relative position encoding class. """
    """ Reference https://github.com/mlpc-ucsd/CoaT/blob/main/src/models/coat.py#L119 """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., shared_crpe=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.dim = dim
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)                                           # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mix_proj = nn.Linear(head_dim*2, head_dim)
        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, mx, size):
        B, N, C = x.shape
        # Generate Q, K, V.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
        mqkv = self.qkv(mx).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
        q, k, v = qkv[0], qkv[1], qkv[2]                                                 # Shape: [B, h, N, Ch].
        mq  = mqkv[0]                                           # Shape: [B, h, N, Ch].

        q = self.mix_proj(torch.cat([mq, q], dim=-1))
        # Factorized attention.
        k_softmax = k.softmax(dim=2)                                                     # Softmax on dim N.
        k_softmax_T_dot_v = einsum('b h n k, b h n v -> b h k v', k_softmax, v)          # Shape: [B, h, Ch, Ch].
        factor_att        = einsum('b h n k, b h k v -> b h n v', mq, k_softmax_T_dot_v)  # Shape: [B, h, N, Ch].

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)                                                # Shape: [B, h, N, Ch].

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)                                           # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].
        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x   # Shape: [B, N, C].




class ConvPosEnc(nn.Module):
    """ Convolutional Position Encoding.
        Note: This module is similar to the conditional position encoding in CPVT. 
        Reference https://github.com/mlpc-ucsd/CoaT/blob/main/src/models/coat.py#L161
    """
    def __init__(self, dim, k=3):
        """init function"""
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        """foward function"""
        B, N, C = x.shape
        H, W = size

        feat = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x
    
class BasicBlock(nn.Module):
    """ Serial block class.
        Note: In this implementation, each serial block only contains a conv-attention and a FFN (MLP) module. """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 shared_cpe=None, shared_crpe=None):
        super().__init__()

        # Conv-Attention.
        self.cpe = shared_cpe

        self.norm1 = norm_layer(dim)
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            shared_crpe=shared_crpe)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP.
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, size):
        # Conv-Attention.
        x = self.cpe(x, size)                  # Apply convolutional position encoding.
        cur = self.norm1(x)
        cur = self.factoratt_crpe(cur, size)   # Apply factorized attention and convolutional relative position encoding.
        x = x + self.drop_path(cur) 

        # MLP. 
        cur = self.norm2(x)
        cur = self.mlp(cur)
        x = x + self.drop_path(cur)

        return x

class BasicMixBlock(nn.Module):
    """ Serial block class.
        Note: In this implementation, each serial block only contains a conv-attention and a FFN (MLP) module. """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 shared_cpe=None, shared_crpe=None):
        super().__init__()

        # Conv-Attention.
        self.cpe = shared_cpe

        self.norm1 = norm_layer(dim)
        self.factoratt_crpe = FactorAtt_Mix_ConvRelPosEnc(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            shared_crpe=shared_crpe)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP.
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mx, size):
        # Conv-Attention.
        x = self.cpe(x, size)                  # Apply convolutional position encoding.
        mx = self.cpe(mx, size)                  # Apply convolutional position encoding.

        cur = self.norm1(x)
        mcur = self.norm1(mx)

        cur = self.factoratt_crpe(cur, mcur, size)   # Apply factorized attention and convolutional relative position encoding.
        x = x + self.drop_path(cur) 

        # MLP. 
        cur = self.norm2(x)
        cur = self.mlp(cur)
        x = x + self.drop_path(cur)

        return x
    
class BasicLayer(nn.Module):
    def __init__(self, depth, in_dim, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., crpe_window={3:2, 5:3, 7:3}, isPool=False):
        super().__init__()
        self.indim = in_dim
        self.dim = dim

        self.path_embed = TransPatchEmbed(in_dim=in_dim, embed_dim=dim, isPool=isPool)
        self.cpe = ConvPosEnc(dim=dim, k=3)
        self.crpe = ConvRelPosEnc(Ch=dim//num_heads, h=num_heads, window=crpe_window)

        self.in_blocks = BasicBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop, shared_cpe=self.cpe, shared_crpe=self.crpe)
        self.blocks = nn.ModuleList([BasicBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                  drop=drop, attn_drop=attn_drop, shared_cpe=self.cpe, shared_crpe=self.crpe) 
                                    for _ in range(depth-1)])
        

    def forward(self, x):
        ps = self.path_embed(x)

        _, _, H, W = ps.shape
        ps = ps.flatten(2).transpose(1, 2).contiguous()    # [B, C, H, W] -> [B, N, C]
        xs = self.in_blocks(ps, size=(H, W))
        for blk in self.blocks:
            xs = blk(xs, size=(H, W))
        out = xs.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
        return out

class BasicMixLayer(nn.Module):
    def __init__(self, depth, in_dim, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., crpe_window={3:2, 5:3, 7:3}, isPool=False):
        super().__init__()
        self.indim = in_dim
        self.dim = dim
        self.depth = depth

        self.path_embed = TransPatchEmbed(in_dim=in_dim, embed_dim=dim, isPool=isPool)
        self.cpe = ConvPosEnc(dim=dim, k=3)
        self.crpe = ConvRelPosEnc(Ch=dim//num_heads, h=num_heads, window=crpe_window)

        # self.in_blocks = BasicBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                             drop=drop, attn_drop=attn_drop, shared_cpe=self.cpe, shared_crpe=self.crpe)
        self.blocks = nn.ModuleList([BasicBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                  drop=drop, attn_drop=attn_drop, shared_cpe=self.cpe, shared_crpe=self.crpe) 
                                    for _ in range(depth-1)])
        self.out_blocks = BasicMixBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop, shared_cpe=self.cpe, shared_crpe=self.crpe)
        

    def forward(self, x, mx):

        mx = F.interpolate(mx, size=x.size()[2:], mode='bilinear', align_corners=False)

        ps = self.path_embed(x)
        mps = self.path_embed(mx)

        _, _, H, W = ps.shape
        xs = ps.flatten(2).transpose(1, 2).contiguous()    # [B, C, H, W] -> [B, N, C]
        xps = mps.flatten(2).transpose(1, 2).contiguous()    # [B, C, H, W] -> [B, N, C]

        for blk in self.blocks:
            xs = blk(xs, size=(H, W))
        xs = self.out_blocks(xs, xps, size=(H, W))

        out = xs.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
        return out


"""
Channel and Pixel Attention Layer.
Reference: https://github.com/trentqq/SGUIE-Net_Simple/blob/master/models/networks.py
"""
# class ChannelAttentionLayer(nn.Module):
#     def __init__(self, channel, reduction=8, bias=True):
#         super(ChannelAttentionLayer, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y
    
class ChannelAttentionLayer(nn.Module):
    """ https://github.com/shiningZZ/GUPDM/blob/main/models/GUPDM.py """
    def __init__(self, channel, reduction=8, bias=True):
        super(ChannelAttentionLayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_y = self.avg_pool(x)
        avg_y = self.conv_du(avg_y)
        max_y = self.max_pool(x)
        max_y = self.conv_du(max_y)
        y = self.sigmoid(avg_y + max_y)

        return x * y

class PixelAttentionLayer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(PixelAttentionLayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, 1, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

class FeatureAttentionModule(nn.Module):
    def __init__(self, net_depth, in_dim, dim, kernel_size):
        super(FeatureAttentionModule, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = Conv(net_depth, in_dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = Conv(net_depth, dim, dim, kernel_size, bias=True)
        self.calayer = ChannelAttentionLayer(dim)
        self.palayer = PixelAttentionLayer(dim)

    def forward(self, x):
        x = self.bn1(x)
        res = self.conv1(x)
        res = self.act1(res)
        res = self.conv2(res)       # From: Attention Network for Non-Uniform Deblurring

        res = self.calayer(res)
        res = self.palayer(res)
        res = res + x
        return res

'''
Detail Repair Branch, From: HCLR-Net: Hybrid Contrastive Learning Regularization with Locally Randomized Perturbation for Underwater Image Enhancement
'''
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(ResidualBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=(kernel_size // 2), bias=True),
            nn.GELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size, padding=(kernel_size // 2), bias=True),
            ChannelAttentionLayer(out_channel, reduction=16),
            PixelAttentionLayer(out_channel, reduction=16)
        )
    def forward(self, x):
        return self.residual(x) + x

# class SAFusion(nn.Module):
#     def __init__(self, net_depth, dim, sa_dim):
#         super(SAFusion, self).__init__()

#         self.conv = ConvLayer(net_depth=net_depth, in_dim=sa_dim, dim=dim, kernel_size=3)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, feat, sa_feat):
#         sa_feat = F.interpolate(sa_feat, size=feat.size()[2:], mode='bilinear', align_corners=False)
#         sa_feat = self.conv(sa_feat)
#         sa_feat = self.softmax(sa_feat)

#         out = feat * sa_feat
#         out = feat + out

#         return out

class SAFusion(nn.Module):
    def __init__(self, dim, sa_dim, m=-0.80):
        super(SAFusion, self).__init__()

        #self.conv = ConvLayer(net_depth=net_depth, in_dim=sa_dim, dim=dim, kernel_size=3)
        self.conv = Conv(sa_dim, dim, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)

        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.w = torch.nn.Parameter(w, requires_grad=True)
        self.mix_block = nn.Sigmoid()


    def forward(self, feat, sa_feat):
        sa_feat = F.interpolate(sa_feat, size=feat.size()[2:], mode='bilinear', align_corners=False)
        sa_feat = self.conv(sa_feat)

        mix_factor = self.mix_block(self.w)
        out = feat * mix_factor.expand_as(feat) + sa_feat * (1 - mix_factor.expand_as(sa_feat))

        # sa_feat = self.softmax(sa_feat)
        # out = feat * sa_feat
        # out = feat + out

        return out
    


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')
        
        self.down = nn.MaxPool2d(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

        self.up = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.up(x)
        return x


class Fusion(nn.Module):
    def __init__(self, dim):
        super(Fusion, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True)
        )


    def forward(self, in_feats):
        out = torch.cat(in_feats, dim=1)
        out = self.proj(out)

        return out



class UNet(nn.Module):
    def __init__(self, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=Fusion):
        super(UNet, self).__init__()

        self.stage_num = 5
        depths = [1, 1, 1, 1, 1, 1, 1]
        embed_dims = [32, 64, 128, 256, 128, 64, 32]

        # input convolution
        self.inconv = ConvLayer(in_dim=3, dim=embed_dims[0], kernel_size=3)

        # encoder layers
        self.layer1 = BasicLayer(in_dim=embed_dims[0], dim=embed_dims[0], depth=depths[0])
        self.down1 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(in_dim=embed_dims[1], dim=embed_dims[1], depth=depths[1])
        self.down2 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(in_dim=embed_dims[2], dim=embed_dims[2], depth=depths[2])
        self.down3 = PatchEmbed(patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.skip3 = nn.Conv2d(embed_dims[2], embed_dims[2], 1)

        # middle convs
        self.middle = BasicLayer(in_dim=embed_dims[3], dim=embed_dims[3], depth=depths[3])
        
        # decoder layers
        self.up4 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])
        self.layer4 = BasicMixLayer(in_dim=embed_dims[4], dim=embed_dims[4], depth=depths[4])
        self.fusion4 = fusion_layer(embed_dims[4])
        
        self.up5 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[5], embed_dim=embed_dims[4])
        self.fusion5 = fusion_layer(embed_dims[5])
        self.layer5 = BasicLayer(in_dim=embed_dims[5], dim=embed_dims[5], depth=depths[5])
        
        self.up6 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[6], embed_dim=embed_dims[5])
        self.layer6 = BasicLayer(in_dim=embed_dims[6], dim=embed_dims[6], depth=depths[6])
        self.fusion6 = fusion_layer(embed_dims[6])

        # output convolution
        self.outconv = ConvLayer(in_dim=embed_dims[6], dim=3, mid_dim=embed_dims[6], kernel_size=3)


    def forward(self, x):
        identity = x
        skips = []

        feat0 = self.inconv(x)

        feat1 = self.layer1(feat0)
        skips.append(self.skip1(feat1))          # skip1: torch.Size([16, 32, 256, 256])
        feat1 = self.down1(feat1)             # feat1: torch.Size([16, 64, 128, 128])
        
        feat2 = self.layer2(feat1)
        skips.append(self.skip2(feat2))          # skip2: torch.Size([16, 64, 128, 128])
        feat2 = self.down2(feat2)             # feat2: torch.Size([16, 128, 64, 64])
  
        feat3 = self.layer3(feat2)
        skips.append(self.skip3(feat3))          # skip3: torch.Size([16, 128, 64, 64])
        feat3 = self.down3(feat3)             # feat3: torch.Size([16, 256, 32, 32])

        featmid = self.middle(feat3)              # feat4: torch.Size([16, 512, 16, 16])

        feat4 = self.up4(featmid)            # feat6: torch.Size([16, 128, 64, 64])
        feat4 = self.fusion4([feat4, skips[2]])
        feat4 = self.layer4(feat4)

        feat5 = self.up5(feat4)            # feat7: torch.Size([16, 64, 128, 128])
        feat5 = self.fusion5([feat5, skips[1]])
        feat5 = self.layer5(feat5)

        feat6 = self.up6(feat5)            # feat8: torch.Size([16, 32, 256, 256])
        feat6 = self.fusion6([feat6, skips[0]])
        feat6 = self.layer6(feat6)

        x = self.outconv(feat6) + identity    # x: torch.Size([16, 3, 256, 256])

        return x

class UNet_SA(nn.Module):
    def __init__(self, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=Fusion):
        super(UNet_SA, self).__init__()

        self.stage_num = 5
        depths = [1, 1, 1, 1, 1, 1, 1]
        embed_dims = [32, 64, 128, 256, 128, 64, 32]

        # input convolution
        self.inconv = ConvLayer(in_dim=3, dim=embed_dims[0], kernel_size=3)

        # encoder layers
        self.layer1 = BasicLayer(in_dim=embed_dims[0], dim=embed_dims[0], depth=depths[0])
        self.down1 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(in_dim=embed_dims[1], dim=embed_dims[1], depth=depths[1])
        self.down2 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(in_dim=embed_dims[2], dim=embed_dims[2], depth=depths[2])
        self.down3 = PatchEmbed(patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.skip3 = nn.Conv2d(embed_dims[2], embed_dims[2], 1)

        # middle convs
        self.middle = BasicMixLayer(in_dim=embed_dims[3], dim=embed_dims[3], depth=depths[3])
        
        # decoder layers
        self.up4 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])
        self.layer4 = BasicMixLayer(in_dim=embed_dims[4], dim=embed_dims[4], depth=depths[4])
        self.fusion4 = fusion_layer(embed_dims[4])
        
        self.up5 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[5], embed_dim=embed_dims[4])
        self.fusion5 = fusion_layer(embed_dims[5])
        self.layer5 = BasicMixLayer(in_dim=embed_dims[5], dim=embed_dims[5], depth=depths[5])
        
        self.up6 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[6], embed_dim=embed_dims[5])
        self.layer6 = BasicLayer(in_dim=embed_dims[6], dim=embed_dims[6], depth=depths[6])
        self.fusion6 = fusion_layer(embed_dims[6])

        self.layer7 = BasicLayer(in_dim=embed_dims[6], dim=embed_dims[6], depth=1)


        # output convolution
        self.outconv = ConvLayer(in_dim=embed_dims[6], dim=3, mid_dim=embed_dims[6], kernel_size=3)


    def forward(self, x, fs):
        identity = x
        skips = []
        up1, up2, up3, up4, up5 = fs["C1"], fs["C2"], fs["C3"], fs["C4"], fs["C5"]

        feat0 = self.inconv(x)

        feat1 = self.layer1(feat0)
        skips.append(self.skip1(feat1))          # skip1: torch.Size([16, 32, 256, 256])
        feat1 = self.down1(feat1)             # feat1: torch.Size([16, 64, 128, 128])
        
        feat2 = self.layer2(feat1)
        skips.append(self.skip2(feat2))          # skip2: torch.Size([16, 64, 128, 128])
        feat2 = self.down2(feat2)             # feat2: torch.Size([16, 128, 64, 64])
  
        feat3 = self.layer3(feat2)
        skips.append(self.skip3(feat3))          # skip3: torch.Size([16, 128, 64, 64])
        feat3 = self.down3(feat3)             # feat3: torch.Size([16, 256, 32, 32])

        featmid = self.middle(feat3, up4)              # feat4: torch.Size([16, 512, 16, 16])

        feat4 = self.up4(featmid)            # feat6: torch.Size([16, 128, 64, 64])
        feat4 = self.fusion4([feat4, skips[2]])
        feat4 = self.layer4(feat4, up3)

        feat5 = self.up5(feat4)            # feat7: torch.Size([16, 64, 128, 128])
        feat5 = self.fusion5([feat5, skips[1]])
        feat5 = self.layer5(feat5, up2)

        feat6 = self.up6(feat5)            # feat8: torch.Size([16, 32, 256, 256])
        feat6 = self.fusion6([feat6, skips[0]])
        feat6 = self.layer6(feat6)

        feat7 = self.layer7(feat6)

        x = self.outconv(feat7) + identity    # x: torch.Size([16, 3, 256, 256])

        return x




class OriginalNet(nn.Module):
    def __init__(self, kernel_size=3):
        super(OriginalNet, self).__init__()

        self.stage_num = 5
        depths = [1, 1, 1, 1, 1]
        net_depth = sum(depths)
        embed_dims = [256, 256, 256, 256, 256]

        # input convolution
        self.inconv = Conv(in_channels=3, out_channels=embed_dims[0], kernel_size=3)

        # backbone layers
        self.block1 = ResidualBlock(in_channel=embed_dims[0], out_channel=embed_dims[0], kernel_size=kernel_size)
        self.block2 = ResidualBlock(in_channel=embed_dims[0], out_channel=embed_dims[1], kernel_size=kernel_size)
        self.block3 = ResidualBlock(in_channel=embed_dims[1], out_channel=embed_dims[2], kernel_size=kernel_size)
        self.block4 = ResidualBlock(in_channel=embed_dims[2], out_channel=embed_dims[3], kernel_size=kernel_size)
        self.block5 = ResidualBlock(in_channel=embed_dims[3], out_channel=embed_dims[4], kernel_size=kernel_size)

        # output convolution
        self.outconv = Conv(in_channels=embed_dims[4], out_channels=3, kernel_size=3)


    def forward(self, x):
        feat = self.inconv(x)

        feat1 = self.block1(feat)
        feat2 = self.block2(feat1)
        feat3 = self.block3(feat2)
        feat4 = self.block4(feat3)

        out = self.outconv(feat+feat4)

        return out

class DTIUIE(nn.Module):
    def __init__(self, kernel_size=3):
        super(DTIUIE, self).__init__()

        self.unet = UNet_SA(kernel_size=kernel_size)
        self.original = OriginalNet(kernel_size=kernel_size)

    def forward(self, x, fs):

        # x = torch.log(x+1/255)*(1 - torch.log(TF.gaussian_blur(x+1/255,kernel_size=7)))    

        out_unet = self.unet(x, fs)
        out_original = self.original(x)
        
        out = out_unet + out_original
        out = normalize_img(out)

        return out

if __name__ == '__main__':
    model = DTIUIE()
    x = torch.randn((1, 3, 256, 256))

    fs = {
          "C1": torch.randn((1, 32, 256, 256)), 
          "C2": torch.randn((1, 64, 128, 128)), 
          "C3": torch.randn((1, 128, 64, 64)), 
          "C4": torch.randn((1, 256, 32, 32)), 
          "C5": torch.randn((1, 512, 32, 32))
          }
          
    seg, up1, up2, up3, up4, up5 = torch.randn((1, 3, 256, 256)), torch.randn((1, 32, 256, 256)), torch.randn((1, 64, 128, 128)), torch.randn((1, 128, 64, 64)), torch.randn((1, 256, 32, 32)), torch.randn((1, 512, 32, 32))

    y = model(x, fs)
    print(y.size())
