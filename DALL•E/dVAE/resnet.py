import torch
import torch.nn as nn
from .utils import ModifiedConv2d

# Defines bottleneck for ResNet
class Bottleneck(nn.Module):
    def __init__(
        self,
        in_planes: int,   # Input channels
        out_planes: int,  # Output channels
        architecture: str # Either encoder or decoder (used to define downsampling or upsampling)
    ) -> None:
        super().__init__()

        # Check architecture parameter
        if architecture != 'decoder' and architecture != 'encoder':
            raise ValueError('"architecture" can only have either "decoder" or "encoder" value')

        # Define in, out and hidden planes
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.hidden_planes = self.out_planes // 4

        # Define encoder architecture
        if architecture == 'encoder':
            self.net = nn.Sequential(
                nn.ReLU(),
                ModifiedConv2d(self.in_planes, self.hidden_planes, 3),
                nn.BatchNorm2d(self.hidden_planes),
                nn.ReLU(),
                ModifiedConv2d(self.hidden_planes, self.hidden_planes, 3),
                nn.BatchNorm2d(self.hidden_planes),
                nn.ReLU(),
                ModifiedConv2d(self.hidden_planes, self.hidden_planes, 3),
                nn.BatchNorm2d(self.hidden_planes),
                nn.ReLU(),
                ModifiedConv2d(self.hidden_planes, self.out_planes, 1),
                nn.BatchNorm2d(self.out_planes)
            )
        # Define decoder architecture
        elif architecture == 'decoder':
            self.net = nn.Sequential(
                nn.ReLU(),
                ModifiedConv2d(self.in_planes, self.hidden_planes, 1),
                nn.BatchNorm2d(self.hidden_planes),
                nn.ReLU(),
                ModifiedConv2d(self.hidden_planes, self.hidden_planes, 3),
                nn.BatchNorm2d(self.hidden_planes),
                nn.ReLU(),
                ModifiedConv2d(self.hidden_planes, self.hidden_planes, 3),
                nn.BatchNorm2d(self.hidden_planes),
                nn.ReLU(),
                ModifiedConv2d(self.hidden_planes, self.out_planes, 3),
                nn.BatchNorm2d(self.out_planes)
            )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # Define identity based on in and out planes
        if self.in_planes != self.out_planes:
            identity = ModifiedConv2d(self.in_planes, self.out_planes, 1)
        else:
            identity = nn.Identity()
        
        out = self.net(x)

        return identity(x) + out


# Defines ResNet architecture to be used in the dVAE decoder
class ResNet(nn.Module):
    def __init__(
        self,
        in_planes: int,
        hidden_planes: int,
        blocks_per_group: int,
        architecture: str  # Either encoder or decoder (used to define downsampling or upsampling)
    ) -> None:
        super().__init__()

        # Check architecture parameter
        if architecture != 'decoder' and architecture != 'encoder':
            raise ValueError('"architecture" can only have either "decoder" or "encoder" value')

        self.in_planes = in_planes
        self.hidden_planes = hidden_planes
        self.blocks_per_group = blocks_per_group
        self.block = Bottleneck
        self.architecture = architecture

        # Define encoder architecture
        if architecture == 'encoder':
            self.layers1 = self._make_block(self.hidden_planes, self.hidden_planes)
            self.layers2 = self._make_block(self.hidden_planes, 2 * self.hidden_planes)
            self.layers3 = self._make_block(2 * self.hidden_planes, 4 * self.hidden_planes)
            self.layers4 = self._make_block(4 * self.hidden_planes, 8 * self.hidden_planes)
            self.sampling = nn.MaxPool2d((2, 2))
        # Define decoder architecture
        elif architecture == 'decoder':
            self.layers1 = self._make_block(self.in_planes, 8 * self.hidden_planes)
            self.layers2 = self._make_block(8 * self.hidden_planes, 4 * self.hidden_planes)
            self.layers3 = self._make_block(4 * self.hidden_planes, 2 * self.hidden_planes)
            self.layers4 = self._make_block(2 * self.hidden_planes, 1 * self.hidden_planes)
            self.sampling = nn.UpsamplingNearest2d(scale_factor=2)

        # Parameters initialization
        for m in self.modules():
            if isinstance(m, ModifiedConv2d):
                # Fills the input Tensor with values according to a normal distribution
                nn.init.kaiming_normal_(tensor=m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # Fills the input Tensor with the value "val"
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)

    def _make_block(self, in_planes: int, hidden_planes: int) -> nn.Sequential:
        """ Defines the building of each layer's blocks """
        # Init and build the layers
        layers = []

        for i in range(0, self.blocks_per_group):
            if i == 0:
                layers.append(Bottleneck(in_planes, hidden_planes, architecture=self.architecture))
            else:
                layers.append(Bottleneck(hidden_planes, hidden_planes, architecture=self.architecture))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass input through the residual blocks
        x = self.layers1(x)
        x = self.sampling(x)
        x = self.layers2(x)
        x = self.sampling(x)
        x = self.layers3(x)
        x = self.sampling(x)
        x = self.layers4(x)
        return x
