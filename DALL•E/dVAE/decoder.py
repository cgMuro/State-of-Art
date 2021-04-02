import torch
import torch.nn as nn
from resnet import ResNet

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
        self.resnet = ResNet(in_planes=in_planes, hidden_planes=hidden_planes, blocks_per_group=blocks_per_group, architecture='decoder')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.input_batch_norm(x)

        x = self.resnet(x)

        x = self.relu(x)

        x = self.output_conv(x)
        x = self.output_batch_norm(x)

        return x
