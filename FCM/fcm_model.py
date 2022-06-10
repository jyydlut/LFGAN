import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
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

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )
    def forward(self, x):
        x = self.mpconv(x)
        return x

class up_d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_d, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                nn.Conv2d(in_ch, in_ch//2, kernel_size=2, padding=0), nn.ReLU(inplace=True))
        self.conv = double_conv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))
        x = torch.cat([x2,x1], dim=1)
        x = self.conv(x)
        return x
class up_h(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_h, self).__init__()
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                nn.Conv2d(in_ch, in_ch//2, kernel_size=2, padding=0), nn.ReLU(inplace=True))
        self.up2 = nn.Conv2d(in_ch, in_ch // 2, kernel_size=2, padding=0)
        self.conv = double_conv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up1(x1)
        x2 = self.up2(x2)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))
        x = torch.cat([x2,x1], dim=1)
        x = self.conv(x)
        return x

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.inc = inconv(in_ch, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 256)

        self.up3 = up_h(256, 256)
        self.up4 = up_d(256, 128)
        self.up5 = up_d(128, 64)
        self.up6 = up_d(64, 32)
        self.outc = nn.Conv2d(32, out_ch, 3, padding=1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up3(x5, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        out = self.outc(x)
        return out
