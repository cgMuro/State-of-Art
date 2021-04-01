import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

# Defines bottleneck for ResNet
class Bottleneck(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int
    ) -> None:
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.hidden_planes = self.out_planes // 4

        self.conv1 = nn.Conv2d(self.in_planes, self.hidden_planes, kernel_size=(1, 1), bias=False)
        self.batch_norm1 = nn.BatchNorm2d(self.hidden_planes)
        self.conv2 = nn.Conv2d(self.hidden_planes, self.hidden_planes, kernel_size=(3, 3), bias=False)
        self.batch_norm2 = nn.BatchNorm2d(self.hidden_planes)
        self.conv3 = nn.Conv2d(self.hidden_planes, self.hidden_planes, kernel_size=(3, 3), bias=False)
        self.batch_norm3 = nn.BatchNorm2d(self.hidden_planes)
        self.conv4 = nn.Conv2d(self.hidden_planes, self.out_planes, kernel_size=(3, 3), bias=False)
        self.batch_norm4 = nn.BatchNorm2d(self.out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if self.in_planes != self.out_planes:
            identity = nn.Conv2d(self.in_planes, self.out_planes, 1)
        else:
            identity = nn.Identity(x)
        out = self.relu(x)

        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.batch_norm3(out)
        out = self.relu(x)

        out = self.conv4(out)
        out = self.batch_norm4(out)
        
        # Add the identity to the output of the last batch normalization layer
        out += identity
        # Pass the result into relu
        out = self.relu(out)

        return out


# Defines ResNet architecture to be used in the dVAE decoder
class ResNet(nn.Module):
    def __init__(
        self,
        in_planes: int,
        hidden_planes: int,
        blocks_per_group: int,
    ) -> None:
        super().__init__()

        self.in_planes = in_planes
        self.hidden_planes = hidden_planes
        self.blocks_per_group = blocks_per_group
        self.block = Bottleneck
        
        self.layers1 = self._make_block(self.in_planes, 8*self.hidden_planes)
        self.layers2 = self._make_block(8*self.hidden_planes, 4*self.hidden_planes)
        self.layers3 = self._make_block(4*self.hidden_planes, 2*self.hidden_planes)
        self.layers4 = self._make_block(2*self.hidden_planes, 1*self.hidden_planes)
        self.upSample = nn.UpsamplingNearest2d(scale_factor=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Fills the input Tensor with values according to a normal distribution
                nn.init.kaiming_normal_(tensor=m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # Fills the input Tensor with the value "val"
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)

    # Define a function to build the layers using either BasicBlock or Bottleneck architectures
    def _make_block(self, in_planes: int, hidden_planes: int) -> nn.Sequential:
        # Init and build the layers
        layers = []

        for i in range(0, self.blocks_per_group):
            if i == 0:
                layers.append(Bottleneck(in_planes, hidden_planes))
            else:
                layers.append(Bottleneck(hidden_planes, hidden_planes))

        return nn.Sequential(*layers)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # Pass input through the residual blocks
        x = self.layers1(x)
        x = self.upSample(x)
        x = self.layers2(x)
        x = self.upSample(x)
        x = self.layers3(x)
        x = self.upSample(x)
        x = self.layers4(x)
        return x

class Decoder(nn.Module):
    def  __init__(
            self,
            in_planes: int,
            hidden_planes: int,
            out_planes: int,
            blocks_per_group: int,
            vocab_size: int = 8192
        ):
        super().__init__()
        self.input_conv = nn.Conv2d(vocab_size, in_planes, kernel_size=(1, 1))
        self.input_batch_norm = nn.BatchNorm2d(in_planes)
        self.output_conv = nn.Conv2d(hidden_planes, 2*out_planes, kernel_size=(1, 1))
        self.output_batch_norm = nn.BatchNorm2d(2*out_planes)
        self.resnet = ResNet(in_planes=in_planes, hidden_planes=hidden_planes, blocks_per_group=blocks_per_group)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.input_batch_norm(x)

        x = self.resnet(x)

        x = self.relu(x)

        x = self.output_conv(x)
        x = self.output_batch_norm(x)

        return x
