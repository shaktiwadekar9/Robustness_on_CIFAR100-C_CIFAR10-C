'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
Reference: https://github.com/ZjjConan/SimAM/blob/master/networks/cifar/mobilenetv2.py
'''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, ou_channels, stride=1):
    return nn.Conv2d(in_channels, ou_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channels, ou_channels, stride=1):
    return nn.Conv2d(in_channels, ou_channels, kernel_size=1, stride=stride, padding=0, bias=False)


class InvertedResidualBlock(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, attention_module=None):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = conv1x1(in_planes, planes, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, out_planes, stride=1)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.relu = nn.ReLU(inplace=True)

        if attention_module is not None:
            if type(attention_module) == functools.partial:
                module_name = attention_module.func.get_module_name()
            else:
                module_name = attention_module.get_module_name()

            if module_name == "simam":
                self.conv2 = nn.Sequential(
                        self.conv2,
                        attention_module(planes)
                )
            else:
                self.bn3 = nn.Sequential(
                        self.bn3,
                        attention_module(out_planes)
                )

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                    conv1x1(in_planes, out_planes, stride=1),
                    nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + self.shortcut(x) if self.stride == 1 else out

        return out



class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, block, num_blocks=0, num_class=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(block, in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_class)

    def _make_layers(self, block, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def MobileNetV2Wrapper(num_class=10, block=None, attention_module=None):

    b = lambda in_planes, out_planes, expansion, stride: \
        InvertedResidualBlock(in_planes, out_planes, expansion, stride, attention_module=attention_module)

    return MobileNetV2(b, num_blocks=0, num_class=num_class)

