"""
From https://github.com/JiaRenChang/PSMNet
Licensed under MIT
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

def conv2d(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation,
            bias=True,
        )
    )

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
                
        self.block1 = nn.Sequential(
            conv2d(3, 16, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(16, 16, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.block2 = nn.Sequential(
            conv2d(16, 32, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(32, 32, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.block3 = nn.Sequential(
            conv2d(32, 64, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(64, 64, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.block4 = nn.Sequential(
            conv2d(64, 96, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(96, 96, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.block5 = nn.Sequential(
            conv2d(96, 128, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(128, 128, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.block6 = nn.Sequential(
            conv2d(128, 192, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(192, 192, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x, mad=False):
        out1 = self.block1(x)
        out2 = self.block2(out1 if not mad else out1.detach())
        out3 = self.block3(out2 if not mad else out2.detach())
        out4 = self.block4(out3 if not mad else out3.detach())
        out5 = self.block5(out4 if not mad else out4.detach())
        out6 = self.block6(out5 if not mad else out5.detach())

        return x, out1, out2, out3, out4, out5, out6

class disparity_decoder(nn.Module):
    def __init__(self, in_channels):
        super(disparity_decoder, self).__init__()
        
        self.decoder = nn.Sequential(
            conv2d(in_channels, 128, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(128, 128, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(128, 96, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(96, 64, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(64, 1, 3, 1, 1, 1),
        )

    def forward(self, x):       
        return self.decoder(x)


class context_net(nn.Module):
    def __init__(self):
        super(context_net, self).__init__()
        
        self.context = nn.Sequential(
            conv2d(33, 128, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(128, 128, 3, 1, 1, 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(128, 128, 3, 1, 1, 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(128, 96, 3, 1, 1, 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(96, 64, 3, 1, 1, 16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(64, 32, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(32, 1, 3, 1, 1, 1),
        )

    def forward(self, x):       
        return self.context(x)