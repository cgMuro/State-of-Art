import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveNet(nn.Module):
    def __init__(
        self,
        channels,
        hidden,
        kernel_size,
        n_blocks
    ):
        super().__init__()

        # Define input layer
        self.input_conv = DilatedCausalConv1d(in_channels=channels, out_channels=hidden, kernel_size=kernel_size)

        # Define main blocks
        self.blocks = nn.ModuleList([])

        for i in range(n_blocks):
            rate = 2**i
            self.blocks.append({
                    # Dilated Causal Convolution
                    'dilated_causal_conv': DilatedCausalConv1d(in_channels=hidden, out_channels=hidden, kernel_size=kernel_size, dilation=rate),
                    # Gated Unit
                    'gated_unit': GatedUnit(),
                    # 1x1 Convolution
                    'conv_1x1': nn.Conv1d(hidden, hidden, (1, 1))
                }
            )
            
        # Define output layers
        self.conv_1x1_1 = nn.Conv1d(hidden, hidden, (1, 1))
        self.relu_1 = nn.ReLU()
        self.conv_1x1_2 = nn.Conv1d(hidden, hidden, (1, 1))
        self.relu_2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Init skip connections
        skips = []

        # Input layer
        x = self.input_conv(x)

        # Main blocks
        for block in self.blocks:
            # Dilated Causal Convolution
            out = block['dilated_causal_conv'](x)
            # Gated Units
            out = block['gated_unit'](out)
            # 1x1 Convolution
            out = block['conv_1x1'](out)
            # Skip connection
            skips.append(out)
            # Residual connection
            x += out

        # Output blocks
        x = functools.reduce((lambda a, b: torch.add(a, b)), skips)
        x = self.relu_1(x)
        x = self.conv_1x1_1(x)
        x = self.relu_2(x)
        x = self.conv_1x1_2(x)
        x = F.softmax(x)

        return x


class DilatedCausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(input, (self.__padding, 0)))



class GatedUnit(nn.Module):
    def __init__(self):
        super().__init__()

        self.tanh = nn.Tanh()
        self.sigmoind = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(self.tanh(x), self.sigmoind(x))
