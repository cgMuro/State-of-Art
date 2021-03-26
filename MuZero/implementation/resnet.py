import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ResidualBlock(nn.Module):
    """ Implementation of basic ResNet residual block """
    def __init__(self, in_planes=3, planes=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3))
        self.batch_norm = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch_norm(out)
        out += x
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """ ResNet implementation """
    def __init__(self, layers: List[int] = [2, 2, 2, 2]) -> None:
        super().__init__()

        # Define layers and functions
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.batch_norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers1 = self._make_layer(in_planes=64, planes=64, blocks=layers[0])
        self.layers2 = self._make_layer(in_planes=64, planes=128, blocks=layers[1], stride=2)
        self.layers3 = self._make_layer(in_planes=128, planes=256, blocks=layers[2], stride=2)
        self.layers4 = self._make_layer(in_planes=256, planes=512, blocks=layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Fills the input Tensor with values according to a normal distribution
                nn.init.kaiming_normal_(tensor=m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # Fills the input Tensor with the value "val"
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)

    # Define a function to build the layers using either BasicBlock or Bottleneck architectures
    def _make_layer(self, in_planes : int, planes : int, blocks : int, stride: int = 1) -> nn.Sequential:
        # Init and build the layers
        layers = [ResidualBlock(in_planes=in_planes, planes=planes) for _ in range(blocks)]
        return nn.Sequential(*layers)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # Pass input into the first block (which is not residual)
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Pass input through the residual blocks
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        # Final Average pooling layer
        x = self.avgpool(x)

        return x
