import torch
import torch.nn as nn
from vgg import VGG

class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.conv_cat = nn.Conv2d(3*out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x_cat = torch.cat((x0, x1, x2), 1)
        x_cat = self.conv_cat(x_cat)
        x = self.relu(x_cat + self.conv_res(x))
        return x

class decoder(nn.Module):
    def __init__(self, channels):
        super(decoder, self).__init__()
        self.convf = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(2*channels, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(3*channels, channels, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)
    def forward(self, x3, x4, x5):
        x3 = self.upsample(x3)
        x4 = self.conv4(torch.cat((x3, x4), dim=1))
        x3 = self.upsample(x3)
        x4 = self.upsample(x4)
        x5_out = self.conv5(torch.cat((x3, x4, x5), dim=1))
        x5 = self.upsample(x5_out)
        x5 = self.upsample(x5)
        out = self.convf(x5)
        return out
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.vgg = VGG()
        self.rfb5 = RFB(512, 32)
        self.rfb4 = RFB(512, 32)
        self.rfb3 = RFB(256, 32)
        self.decoder = decoder(32)
    def forward(self, x):
        x1 = self.vgg.conv1(x)
        x2 = self.vgg.conv2(x1)
        x3 = self.vgg.conv3(x2)
        x4 = self.vgg.conv4_1(x3)
        x5 = self.vgg.conv5_1(x4)
        x5f = self.rfb5(x5)
        x4f = self.rfb4(x4)
        x3f = self.rfb3(x3)
        out = self.decoder(x5f, x4f, x3f)
        out = torch.sigmoid(out)
        return out, [x5f, x4f, x3f]
class modelfocal(nn.Module):
    def __init__(self):
        super(modelfocal, self).__init__()
        self.vgg = VGG()
        self.rfb5 = RFB(512, 32)
        self.rfb4 = RFB(512, 32)
        self.rfb3 = RFB(256, 32)
        self.decoder1 = decoder(32)
        self.decoder2 = decoder(32)
        self.concat5 = nn.Conv2d(13 * 32, 32, 3, padding=1)
        self.concat4 = nn.Conv2d(13 * 32, 32, 3, padding=1)
        self.concat3 = nn.Conv2d(13 * 32, 32, 3, padding=1)
    def cat_rf(self, ffeat, rfeat, mode):
        if mode == 'train':
            f1, f2 = ffeat[:12], ffeat[12:]
            r1, r2 = rfeat[:1], rfeat[1:]
            o1, o2 = torch.cat((f1, r1), dim=0), torch.cat((f2, r2), dim=0)
            o1 = torch.cat(torch.chunk(o1, 13, dim=0), dim=1)
            o2 = torch.cat(torch.chunk(o2, 13, dim=0), dim=1)
            out = torch.cat((o1, o2), dim=0)
            return out
        else:
            out = torch.cat((ffeat, rfeat), dim=0)
            out = torch.cat(torch.chunk(out, 13, dim=0), dim=1)
            return out
    def forward(self, x, rgbfeat, mode='test'):
        x1 = self.vgg.conv1(x)
        x2 = self.vgg.conv2(x1)
        x3 = self.vgg.conv3(x2)
        x4 = self.vgg.conv4_1(x3)
        x5 = self.vgg.conv5_1(x4)
        focalf5 = self.rfb5(x5)
        focalf4 = self.rfb4(x4)
        focalf3 = self.rfb3(x3)
        outf = self.decoder1(focalf5, focalf4, focalf3)
        outf = torch.sigmoid(outf)
        rgbf5, rgbf4, rgbf3 = rgbfeat
        f5, f4, f3 = self.cat_rf(focalf5, rgbf5, mode), self.cat_rf(focalf4, rgbf4, mode), self.cat_rf(focalf3, rgbf3, mode)
        f5, f4, f3 = self.concat5(f5), self.concat4(f4), self.concat3(f3)
        out = self.decoder2(f5, f4, f3)
        out = torch.sigmoid(out)
        return outf, out