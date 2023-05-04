# License: https://github.com/milesial/Pytorch-UNet
""" Full assembly of the parts to form the complete network """

from torch import nn
from UNet.unet_parts import DoubleConv, Down, Up, OutConv
import ipdb

class UNet(nn.Module):
    def __init__(self, n_channels=3, out_channels=2, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        # print('x shape:',x.shape)
        x1 = self.inc(x)
        # print('x1 shape:',x1.shape)
        x2 = self.down1(x1)
        # print('x2 shape:',x2.shape)
        x3 = self.down2(x2)
        # print('x3 shape:',x3.shape)
        x4 = self.down3(x3)
        # print('x4 shape:',x4.shape)
        x5 = self.down4(x4)
        # print('x5 shape:',x5.shape)
        x = self.up1(x5, x4)
        # print('x(x5,x4) shape:',x.shape)
        x = self.up2(x, x3)
        # print('x(x,x3) shape:',x.shape)
        x = self.up3(x, x2)
        # print('x(x,x2) shape:',x.shape)
        x = self.up4(x, x1)
        # print('x(x,x1) shape:',x.shape)
        logits = self.outc(x)
        # print(logits.shape)
        # ipdb.set_trace()
        return logits

    def _clas(self, num_cls):
        clas = nn.Sequential(
            nn.Conv2d(64,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1)
        )
