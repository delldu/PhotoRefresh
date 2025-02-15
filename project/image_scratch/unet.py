"""Model Helper."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 06月 16日 星期三 00:13:22 CST
# ***
# ************************************************************************************/
#
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class Downsample(nn.Module):
    # https://github.com/adobe/antialiased-cnns
    def __init__(self, filt_size=3, stride=2, channels=64, pad_off=0):
        super(Downsample, self).__init__()

        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        # self.pad_sizes == [1, 1, 1, 1]

        self.stride = stride
        self.channels = channels

        a = np.array([1.0, 2.0, 1.0])
        filt = torch.Tensor(a[:, None] * a[None, :])
        # filt -- 3x3
        # tensor([[1., 2., 1.],
        #         [2., 4., 2.],
        #         [1., 2., 1.]])
        filt = filt / torch.sum(filt)
        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = nn.ReflectionPad2d(self.pad_sizes)

    def forward(self, input):
        return F.conv2d(self.pad(input), self.filt, stride=self.stride, groups=input.shape[1])


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        depth=4,
        conv_num=2,
        wf=6,
        padding=True,
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
                in_channels (int): number of input channels
                out_channels (int): number of output channels
                depth (int): depth of the network
                wf (int): number of filters in the first layer is 2**wf
                padding (bool): if True, apply padding such that the input shape
                                                is the same as the output.
                                                This may introduce artifacts
        """
        super().__init__()
        # in_channels = 1
        # out_channels = 1

        # Define max GPU/CPU memory -- 8G, 400ms
        self.MAX_H = 1024
        self.MAX_W = 2048
        self.MAX_TIMES = 16

        self.padding = padding
        self.depth = depth - 1
        prev_channels = in_channels

        self.first = nn.Sequential(
            *[
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels, 2 ** wf, kernel_size=7),
                nn.LeakyReLU(0.2, True),
            ]
        )
        prev_channels = 2 ** wf

        self.down_path = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        for i in range(depth):
            self.down_sample.append(
                nn.Sequential(
                    *[
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(
                            prev_channels,
                            prev_channels,
                            kernel_size=3,
                            stride=1,
                            padding=0,
                        ),
                        nn.BatchNorm2d(prev_channels),
                        nn.LeakyReLU(0.2, True),
                        Downsample(channels=prev_channels, stride=2),
                    ]
                )
            )
            self.down_path.append(UNetConvBlock(conv_num, prev_channels, 2 ** (wf + i + 1), padding))
            prev_channels = 2 ** (wf + i + 1)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_path.append(UNetUpBlock(conv_num, prev_channels, 2 ** (wf + i), padding))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Sequential(
            *[
                nn.ReflectionPad2d(1),
                nn.Conv2d(prev_channels, out_channels, kernel_size=3),
                nn.Tanh(),
            ]
        )
        self.load_weights()

    def load_weights(self, model_path="models/image_scratch.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        self.load_state_dict(torch.load(checkpoint))

    def forward(self, input_tensor):
        # 0.299 R + 0.587 G + 0.114 B
        x = 0.299 * input_tensor[:, 0:1, :, :] + 0.587 * input_tensor[:, 1:2, :, :] + 0.114 * input_tensor[:, 2:3, :, :]

        x = self.first(x)

        blocks = []
        # for i, down_block in enumerate(self.down_path):
        #     blocks.append(x)
        #     x = self.down_sample[i](x)
        #     x = down_block(x)
        for sample_block, down_block in zip(self.down_sample, self.down_path):
            blocks.append(x)
            x = sample_block(x)
            x = down_block(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        y = self.last(x).clamp(0, 1.0)
        one = torch.ones_like(y)
        zero = torch.zeros_like(y)

        mask_tensor = torch.where(y > 0.5, zero, one)
        return torch.cat((input_tensor, mask_tensor), dim=1)


class UNetConvBlock(nn.Module):
    def __init__(self, conv_num, in_size, out_size, padding):
        super(UNetConvBlock, self).__init__()
        block = []

        for _ in range(conv_num):
            block.append(nn.ReflectionPad2d(padding=int(padding)))
            block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=0))
            block.append(nn.BatchNorm2d(out_size))
            block.append(nn.LeakyReLU(0.2, True))
            in_size = out_size

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, conv_num, in_size, out_size, padding):
        super(UNetUpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=0),
        )

        self.conv_block = UNetConvBlock(conv_num, in_size, out_size, padding)

    def center_crop(self, layer, H: int, W: int):
        _, _, height, width = layer.size()
        y1 = (height - H) // 2
        y2 = y1 + H
        x1 = (width - W) // 2
        x2 = x1 + W

        return layer[:, :, y1:y2, x1:x2]

    def forward(self, x, bridge):
        up = self.up(x)
        B, C, H, W = up.shape
        crop1 = self.center_crop(bridge, H, W)
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
