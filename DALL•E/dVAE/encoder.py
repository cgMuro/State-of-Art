import torch
import torch.nn as nn
from resnet import ResNet
from utils import ModifiedConv2d

class Encoder(nn.Module):
    def  __init__(
            self,
            in_planes: int = 3,
            hidden_planes: int = 256,
            out_planes: int = 10,
            blocks_per_group: int = 2,
            vocab_size: int = 8192
        ):
        super().__init__()
        
        # Input convolution and batch normalization
        self.input_conv = ModifiedConv2d(in_planes, hidden_planes, 7)
        self.input_batch_norm = nn.BatchNorm2d(hidden_planes)
        # Output convolution and batch normalization
        self.output_conv = ModifiedConv2d(8 * hidden_planes, vocab_size, 1)
        self.output_batch_norm = nn.BatchNorm2d(vocab_size)
        # ResNet
        self.resnet = ResNet(in_planes=in_planes, hidden_planes=hidden_planes, blocks_per_group=blocks_per_group, architecture='encoder')
        # ReLU
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolution + batch normalization
        x = self.input_conv(x)
        x = self.input_batch_norm(x)

        # ResNet
        x = self.resnet(x)
        # ReLU
        x = self.relu(x)

        # Convolution + batch normalization
        x = self.output_conv(x)
        x = self.output_batch_norm(x)

        return x
