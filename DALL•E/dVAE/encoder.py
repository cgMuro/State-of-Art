import torch
import torch.nn as nn
from resnet import ResNet

class Encoder(nn.Module):
    def  __init__(
            self,
            in_planes: int,
            hidden_planes: int,
            out_planes: int,
            blocks_per_group: int,
            vocab_size: int = 8192
        ):
        super().__init__()
        self.input_conv = nn.Conv2d(in_planes, hidden_planes, kernel_size=(7, 7))
        self.input_batch_norm = nn.BatchNorm2d(hidden_planes)
        self.output_conv = nn.Conv2d(8*hidden_planes, vocab_size, kernel_size=(1, 1))
        self.output_batch_norm = nn.BatchNorm2d(vocab_size)
        self.resnet = ResNet(in_planes=in_planes, hidden_planes=hidden_planes, blocks_per_group=blocks_per_group, architecture='encoder')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.input_batch_norm(x)

        x = self.resnet(x)

        x = self.relu(x)

        x = self.output_conv(x)
        x = self.output_batch_norm(x)

        return x
