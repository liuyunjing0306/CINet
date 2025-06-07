import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
import numpy as np

from torch.autograd import Variable

# from util import get_alphabet

torch.set_printoptions(precision=None, threshold=1000000, edgeitems=None, linewidth=None, profile=None)


# alphabet = get_alphabet()
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, downsample):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample != None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, num_in, block, layers):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(num_in, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2, 2), (2, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.layer1_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer1 = self._make_layer(block, 128, 256, layers[0])
        self.layer1_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer1_bn = nn.BatchNorm2d(256)
        self.layer1_relu = nn.ReLU(inplace=True)

        self.layer2_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer2 = self._make_layer(block, 256, 512, layers[1])
        self.layer2_conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.layer2_bn = nn.BatchNorm2d(512)
        self.layer2_relu = nn.ReLU(inplace=True)

        self.layer3_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer3 = self._make_layer(block, 512, 1024, layers[2])
        self.layer3_conv = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.layer3_bn = nn.BatchNorm2d(1024)
        self.layer3_relu = nn.ReLU(inplace=True)

        self.layer4_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer4 = self._make_layer(block, 512, 512, layers[3])
        self.layer4_conv2 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.layer4_conv2_bn = nn.BatchNorm2d(1024)
        self.layer4_conv2_relu = nn.ReLU(inplace=True)

    def _make_layer(self, block, inplanes, planes, blocks):

        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 3, 1, 1),
                nn.BatchNorm2d(planes), )
        else:
            downsample = None
        layers = []
        layers.append(block(inplanes, planes, downsample))
        for i in range(1, blocks):
            layers.append(block(planes, planes, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        out1 = self.relu1(x)

        x = self.pool(out1)
        x = self.conv2(x)
        x = self.bn2(x)
        out2 = self.relu2(x)

        x = self.layer1_pool(out2)
        x = self.layer1(x)
        x = self.layer1_conv(x)  #
        x = self.layer1_bn(x)
        out3 = self.layer1_relu(x)

        x = self.layer2_pool(out3)
        x = self.layer2(x)
        x = self.layer2_conv(x)
        x = self.layer2_bn(x)
        out4 = self.layer2_relu(x)

        x = self.layer3_pool(out4)
        x = self.layer3(x)
        x = self.layer3_conv(x)
        x = self.layer3_bn(x)
        out5 = self.layer3_relu(x)

        return out1, out2, out3, out4, out5



