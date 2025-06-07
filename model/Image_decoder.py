import torch.nn as nn
import torch
import math, copy, time
import pdb
import torch.nn.functional as F
from .Backbone import ResNet, BasicBlock


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class CAM(nn.Module):
    def __init__(self, hchannel, channel):
        super(CAM, self).__init__()
        self.conv1_1 = Conv1x1(hchannel + channel, channel)
        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=2)
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=3)
        self.dconv9_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=4)
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = ConvBNR(channel, channel, 3)

    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf, hf), dim=1)
        x = self.conv1_1(x)
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.dconv5_1(xc[1] + x0 + xc[2])
        x2 = self.dconv7_1(xc[2] + x1 + xc[3])
        x3 = self.dconv9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.conv3_3(x + xx)

        return x


class Image_Decoder(nn.Module):
    def __init__(self):
        super(Image_Decoder, self).__init__()
        self.cam1 = CAM(128, 64)
        self.cam2 = CAM(256, 128)
        self.cam3 = CAM(512, 256)
        self.cam4 = CAM(1024, 512)
        self.predictor1 = nn.Conv2d(64, 3, 1)
        self.predictor2 = nn.Conv2d(128, 3, 1)
        self.predictor3 = nn.Conv2d(256, 3, 1)
        self.predictor4 = nn.Conv2d(512, 3, 1)

    def forward(self, out1, out2, out3, out4, out5):
        x45 = self.cam4(out4, out5)
        x345 = self.cam3(out3, x45)
        x2345 = self.cam2(out2, x345)
        x12345 = self.cam1(out1, x2345)

        o4 = self.predictor4(x45)
        o4 = F.interpolate(o4, scale_factor=8, mode='bilinear', align_corners=False)
        o3 = self.predictor3(x345)
        o3 = F.interpolate(o3, scale_factor=4, mode='bilinear', align_corners=False)
        o2 = self.predictor2(x2345)
        o2 = F.interpolate(o2, scale_factor=2, mode='bilinear', align_corners=False)
        o1 = self.predictor1(x12345)
        # o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        return o4, o3, o2, o1


