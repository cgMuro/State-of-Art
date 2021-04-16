import torch
import torch.nn as nn
from resnet import ResNet
from utils import ModifiedConv2d

class Decoder(nn.Module):
    def  __init__(
            self,
            in_planes: int = 128,        # Input channels
            hidden_planes: int = 256,    # Hidden units
            out_planes: int = 3,         # Output channels
            blocks_per_group: int = 2,   # Number of ResNet's bottleneck blocks for each group
            vocab_size: int = 8192       # Size of vocaboulary
        ):
        super().__init__()
        # Input convolution and batch normalization
        self.input_conv = ModifiedConv2d(vocab_size, in_planes, 1)
        self.input_batch_norm = nn.BatchNorm2d(in_planes)
        # Output convolution and batch normalization
        self.output_conv = ModifiedConv2d(hidden_planes, out_planes, 1)
        self.output_batch_norm = nn.BatchNorm2d(out_planes)
        # ResNet
        self.resnet = ResNet(in_planes=in_planes, hidden_planes=hidden_planes, blocks_per_group=blocks_per_group, architecture='decoder')
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
