"""
implementation of UNet for medicine image segmentation
downsample 4 times, upsample 4 times, they concatnate (in channels dim) together(copy and crop), not add directly
input image size: width-1632, height-1216
use conv3x3, conv2x2, conv1x1

Attention:
  The original paper dosen't use padding, so the feature map size will decrease 2 after one convolution
  But in this code, we use padding = 1 to keep the size

paper:
  https://arxiv.org/abs/1505.04597

Based on:
  https://github.com/JavisPeng/u_net_liver

Future work:
  Make the upsampling could be learned too.(but it need more GPU memory) TODO
  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
  reference: https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
            x = self.conv(x)
            return x


class UNet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(UNet, self).__init__()
        
        # downsample
        # in_ch = 3 for default RGB image
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        # upsample
        # out_ch represents the n_classes
        # in our dataset, we set the n_classes = 1
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1)

    """
    my GPU is 1080ti, however it cannot run under the channel[64--128--256--512--1024]
    even I use the same variable to save the memory,  still has the out of memory condition
    and I tried the channel[16--32--64--128--256], it can work
    maybe it's a engineering problem that need to optimize TODO

    """
    def forward(self,x):
        x1 = self.conv1(x)

        x2 = self.pool1(x1)
        x2 = self.conv2(x2)

        x3 = self.pool2(x2)
        x3 = self.conv3(x3)

        x4 = self.pool3(x3)
        x4 = self.conv4(x4)

        x5 = self.pool4(x4)
        x5 = self.conv5(x5)

        x = self.up6(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv6(x)

        x = self.up7(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv7(x)

        x = self.up8(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv8(x)

        x = self.up9(x)
        x = torch.cat([x,x1],dim=1)
        x = self.conv9(x)

        x = self.conv10(x)
        #out = nn.Sigmoid()(c10)
        return x