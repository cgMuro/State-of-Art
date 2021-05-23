import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import mu_law_decoding


class WaveNet(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden,
        kernel_size,
        n_blocks
    ):
        super().__init__()

        self.num_classes = num_classes

        # Define input layer
        self.input_conv = DilatedCausalConv1d(in_channels=num_classes, out_channels=hidden, kernel_size=kernel_size)

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
        self.conv_1x1_2 = nn.Conv1d(hidden, num_classes, (1, 1))
        self.relu_2 = nn.ReLU()

    def generate(
        self,
        num_samples: int = 10,  # Number of samples to generate
        first_samples = None    # Starting samples
    ):
        # Model in prediction mode
        self.eval()

        # Create first sample if needed
        if first_samples is None:
            first_samples = torch.zeros(1) + (self.num_classes // 2)

        # Get to number of samples
        num_given_samples = first_samples.size(0)

        # Init input
        input = torch.zeros(1, self.num_classes, 1)
        # Scatter input and reshape
        input = input.scatter_(1, first_samples[0:1].view(1, -1, 1), 1.)

        for sample in range(num_given_samples - 1):
            # Get prediction from model
            output = self.forward(input)
            
            # Zero out input
            input.zero_()
            # Scatter input and reshape
            input = input.scatter_(1, first_samples[sample+1:sample+2].view(1, -1, 1), 1.).view(1, self.num_classes, 1)


        # Generate a new sample

        # Init generated samples array
        generated = np.array([])
        # Init regularizer
        regularizer = torch.pow(torch.arange(self.num_classes) - self.num_classes / 2., 2)
        regularizer = regularizer.squeeze() * regularizer

        for sample in range(num_samples):
            # Get prediction from model
            output = self.forward(input).squeeze()
            # Regularize output
            output -= regularizer
            
            # Get softmax probabilities
            prob = F.softmax(output, dim=0)
            prob = prob.data.numpy()
            # Generate a random sample from self.num_classes with the associated probabilities prob
            out = np.random.choice(self.num_classes, p=prob)
            out = np.array([out])

            # Update array of generated samples
            generated = np.append(
                generated, 
                (out / self.num_classes) * 2. - 1
            )

            out = torch.from_numpy(out)

            # Zero out input
            input.zero_()
            # Scatter input and reshape
            input = input.scatter_(1, out.view(1, -1, 1), 1.).view(1, self.num_classes, 1)

        # Decode the generated samples and return them
        return mu_law_decoding(generated, self.num_classes)

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
