# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM
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
class OurFuseUperHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6),
                 in_channels_all=[2048, 1024, 512, 256],
                 out_channels_all=[256, 128, 64, 32],
                 upsample_log=[3, 2, 1, 2],
                 per_emb_nums=[16, 8, 4, 4],
                 supervises=[True, True, True, True],
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.in_channels_all = in_channels_all
        self.out_channels_all = out_channels_all
        self.upsample_log = upsample_log
        self.per_emb_nums = per_emb_nums
        self.supervises = supervises
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            (len(self.in_channels)) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.pre_interpolate = nn.ModuleList([nn.Sequential(
            nn.Conv3d(self.out_channels_all[3-i], self.out_channels_all[3-i], 3, 1, 1, 1, self.out_channels_all[3-i]),
            nn.Conv3d(self.out_channels_all[3-i], self.channels, 1),
            nn.BatchNorm3d(self.channels),
            nn.ReLU()
        ) for i in range(0, len(self.out_channels_all)-1)])
        self.post_fuse = nn.ModuleList([nn.Sequential(*[ConvBlock(self.channels) for _ in range(2)]) for o in self.out_channels_all[:]])
        self.kp_encoder = nn.ModuleList([nn.Sequential(
            nn.Conv3d(self.channels, self.per_emb_nums[-1] * self.num_classes, 1),
            # nn.BatchNorm2d(per_emb_num * num_joints),
            nn.ReLU()
        )])
        num = len(self.out_channels_all)
        self.converge = nn.ModuleList([ConvergeHead(self.per_emb_nums[-1], 2 ** self.upsample_log[-1], 3,
                                                    1, self.num_classes)])
        self.fuse_two = nn.ModuleList(
            [nn.Sequential(*[ConvBlock(self.channels) for _ in range(2)])])


    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)
        laterals = []
        for i in range(len(inputs) - 1):
            laterals.append(self.pre_interpolate[i](inputs[i]))
        # build laterals
        #laterals = [
            #lateral_conv(inputs[i])
            #for i, lateral_conv in enumerate(self.lateral_convs)
        #]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='trilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = self.post_fuse[i](fpn_outs[i])
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='trilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)


        #feats = self.fpn_bottleneck(fpn_outs[0])
        #feats = self.fuse_two[-1](feats)


        kp_emb = self.kp_encoder[-1](feats)
        hp = self.converge[-1](kp_emb)
        return hp

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        #output = self.cls_seg(output)
        return output
