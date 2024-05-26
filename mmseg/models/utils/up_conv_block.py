# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_upsample_layer

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

        self.conv = nn.Conv3d(in_dim*num_joints, (up_ratio**2)*num_joints,
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

class UpConvBlockUNet(nn.Module):
    """Upsample convolution block in decoder for UNet.

    This upsample convolution block consists of one upsample module
    followed by one convolution block. The upsample module expands the
    high-level low-resolution feature map and the convolution block fuses
    the upsampled high-level low-resolution feature map and the low-level
    high-resolution feature map from encoder.

    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
        high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block.
            Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv'). If the size of
            high-level feature map is the same as that of skip feature map
            (low-level feature map from encoder), it does not need upsample the
            high-level feature map and the upsample_cfg is None.
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 conv_block,
                 in_channels,
                 skip_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 dcn=None,
                 plugins=None):
        super().__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.conv_block = conv_block(
            in_channels=2 * skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dcn=None,
            plugins=None)
        if upsample_cfg is not None:
            self.upsample = build_upsample_layer(
                cfg=upsample_cfg,
                in_channels=in_channels,
                out_channels=skip_channels,
                with_cp=with_cp,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.upsample = ConvModule(
                in_channels,
                skip_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, skip, x):
        """Forward function."""

        x = self.upsample(x)
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)

        return out
class UpConvBlock(nn.Module):
    """Upsample convolution block in decoder for UNet.

    This upsample convolution block consists of one upsample module
    followed by one convolution block. The upsample module expands the
    high-level low-resolution feature map and the convolution block fuses
    the upsampled high-level low-resolution feature map and the low-level
    high-resolution feature map from encoder.

    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
        high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block.
            Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv'). If the size of
            high-level feature map is the same as that of skip feature map
            (low-level feature map from encoder), it does not need upsample the
            high-level feature map and the upsample_cfg is None.
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 conv_block,
                 in_channels,
                 skip_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 dcn=None,
                 plugins=None):
        super().__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.conv_block = conv_block(
            in_channels=2 * skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dcn=None,
            plugins=None)
        if upsample_cfg is not None:
            self.upsample = build_upsample_layer(
                cfg=upsample_cfg,
                in_channels=in_channels,
                out_channels=skip_channels,
                with_cp=with_cp,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.kp_encoderfinal = nn.Sequential(
                nn.Conv3d(in_channels, in_channels * 2, 1),
                # nn.BatchNorm2d(per_emb_num * num_joints),
                nn.ReLU()
            )

            self.convergefinal3 = ConvergeHead(1, 2 ** 1, 3,
                                                             1, in_channels)
            self.convergefinal5 = ConvergeHead(1, 2 ** 1, 5,
                                              2, in_channels)
            self.convergefinal7 = ConvergeHead(1, 2 ** 1, 7,
                                              3, in_channels)
        else:
            self.kp_encoderfinal = nn.Sequential(
                nn.Conv3d(in_channels, in_channels * 2, 1),
                # nn.BatchNorm2d(per_emb_num * num_joints),
                nn.ReLU()
            )
            self.convergefinal3 = ConvergeHead(1, 2 ** 1, 3,
                                                             1, in_channels)
            self.convergefinal5 = ConvergeHead(1, 2 ** 1, 5,
                                              2, in_channels)
            self.convergefinal7 = ConvergeHead(1, 2 ** 1, 7,
                                              3, in_channels)

            self.upsample = ConvModule(
                in_channels,
                skip_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, skip, x):
        """Forward function."""

        #x = self.upsample(x)

        i = int(x.shape[1] / 64)
        if(i < 9):#9
            if(i<9):
                x = self.convergefinal5(x)
            else:
                x = self.convergefinal3(x)

        else:
            x = self.upsample(x)

        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)

        return out
