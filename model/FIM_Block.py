## 2024.9.19 特征交互模块
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Conv3x3(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class FIM(nn.Module):
    def __init__(self):
        super(FIM, self).__init__()
        self.reduce1 = nn.Sequential(ConvBNR(1024, 256, 3),
                                     ConvBNR(256, 256, 3),
                                     ConvBNR(256, 1024, 3))
        self.reduce2 = nn.Sequential(ConvBNR(1024, 256, 3),
                                     ConvBNR(256, 256, 3),
                                     ConvBNR(256, 1024, 3))
        self.common = Conv3x3(2048, 1024)

    def forward(self, x1, x2):
        x1 = self.reduce1(x1)
        x2 = self.reduce2(x2)
        out = torch.cat((x1, x2), dim=1)
        out = self.common(out)
        return x1, x2, out    ## x1 修复支



