# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from ..utils import resize

class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class ConvergeHead(nn.Module):
    def __init__(self, in_dim, up_ratio, kernel_size, padding, num_joints):
        super().__init__()
        self.in_dim = in_dim
        self.up_ratio = up_ratio
        self.num_joints = num_joints

        self.conv = nn.Conv3d(in_dim*num_joints, (up_ratio**3)*num_joints,
            kernel_size, 1, padding, 1, num_joints)
        self.apply(self._init_weights)

    def forward(self, x):
        hp = self.conv(x)
        #hp = F.pixel_shuffle(hp, self.up_ratio)
        poxel = PixelShuffle3d(self.up_ratio)
        hp = poxel(hp)

        return hp
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class ConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.BatchNorm3d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3) # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x

@MODELS.register_module()
class SRPoseHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 in_channels_all=[2048, 1024, 512, 256],
                 out_channels_all=[256, 128, 64, 32],
                 extra=None,
                 upsample_log=[3, 2, 1, 2],
                 per_emb_nums=[16, 8, 4, 4],
                 supervises=[True, True, True, True],


                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.in_channels_all = in_channels_all
        self.out_channels_all = out_channels_all
        self.upsample_log = upsample_log
        self.per_emb_nums = per_emb_nums
        self.supervises = supervises
        super().__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        self.lr_head = nn.Sequential(
            nn.Conv3d(self.in_channels_all[0], self.num_classes, 1),
            nn.ReLU()
        )
        self.lr_fuse = nn.Sequential(
            nn.Conv3d(self.num_classes, self.out_channels_all[0], 1),
            nn.ReLU()
        )

        self.pre_interpolate = nn.ModuleList([nn.Sequential(
            nn.Conv3d(self.out_channels_all[i - 1], self.out_channels_all[i - 1], 3, 1, 1, 1, self.out_channels_all[i - 1]),
            nn.Conv3d(self.out_channels_all[i - 1], self.out_channels_all[i], 1),
            nn.BatchNorm3d(self.out_channels_all[i]),
            nn.ReLU()
        ) for i in range(1, len(self.out_channels_all))])
        self.pre_fuse = nn.ModuleList([nn.Sequential(
            nn.Conv3d(i, o, 1),
            nn.BatchNorm3d(o),
            nn.ReLU()
        ) for (i, o) in zip(self.in_channels_all, self.out_channels_all)])
        self.fuse = nn.ModuleList([nn.Conv3d(2 * o, o, 1) for o in self.out_channels_all[1:]])
        self.post_fuse = nn.ModuleList([nn.Sequential(*[ConvBlock(o) for _ in range(2)]) for o in self.out_channels_all[1:]])

        self.kp_encoder = nn.ModuleList([nn.Sequential(
            nn.Conv3d(out_channel, per_emb_num * self.num_classes, 1),
            # nn.BatchNorm2d(per_emb_num * num_joints),
            nn.ReLU()
        ) if supervise else nn.Identity() for out_channel, per_emb_num, supervise in
                                         zip(self.out_channels_all, self.per_emb_nums, self.supervises)])
        num = len(self.out_channels_all)
        self.converge = nn.ModuleList([ConvergeHead(self.per_emb_nums[i], 2 ** self.upsample_log[i], 11 - 2 * (num - i),
                                                    5 - (num - i), self.num_classes) if self.supervises[i] else nn.Identity() for i
                                       in range(num)])


        pool_scales = (1, 2, 3, 6)
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels_all[0],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels_all[0] + len(pool_scales) * self.channels,
            self.in_channels_all[0],
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=3,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
    def _forward_feature(self, x):
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        feats = self.bottleneck(psp_outs)
        return feats

    def _forward_feature_sr(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        #xs = inputs[::-1]
        xs=[]
        heatmaps = []

        for i in range(len(inputs)):
            if i == 0:
                xs.append(self._forward_feature(inputs[-1]))
                for j in range(len(inputs) - 1):
                    xs.append(inputs[len(inputs) - j - 2])
                hp = self.lr_head(xs[0])
                heatmaps.append(hp)
                feat = self.pre_fuse[i](xs[i]) + self.lr_fuse(hp)
            else:
                feat = self.pre_interpolate[i - 1](feat)
                feat = F.interpolate(feat, xs[i].shape[2:])
                feat = torch.cat([feat, self.pre_fuse[i](xs[i])], 1)
                feat = self.fuse[i - 1](feat)
                feat = self.post_fuse[i - 1](feat)
            if self.supervises[i] and self.training:
                kp_emb = self.kp_encoder[i](feat)
                hp = self.converge[i](kp_emb)
                heatmaps.append(hp)
        if self.training:
            return heatmaps
        else:
            kp_emb = self.kp_encoder[-1](feat)
            hp = self.converge[-1](kp_emb)
            return hp
    def forward(self, inputs):
        """Forward function."""
        #output = self._forward_feature(inputs)
        output =   self._forward_feature_sr(inputs)

        #output = self.cls_seg(output)
        return output
