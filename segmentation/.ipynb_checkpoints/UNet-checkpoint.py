'''
author: Doby
dobylive01@gmail.com
'''

import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(out_filters)
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()

        self.convBlk = ConvBlock(in_filters, out_filters)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.convBlk(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.transposeConv = nn.ConvTranspose2d(in_filters, out_filters, kernel_size=2, stride=2)
        self.convBlk = ConvBlock(in_filters, out_filters)
        
    def forward(self, x, skip):
        x = self.transposeConv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.convBlk(x)
        
        return x

class UNet(nn.Module):
    def __init__(self, channel):
        super().__init__()

        # Constracting Path
        self.e1 = EncoderBlock(channel, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        # Bridge
        self.b = ConvBlock(512, 1024)

        # Expanding Path
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        self.convOut = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        b = self.b(p4)
        
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        output = self.convOut(d4)
        output = self.sigmoid(output)
        
        return output