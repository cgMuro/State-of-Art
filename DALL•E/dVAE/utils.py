import math
import typing
import torch
import torch.nn as nn
from torchvision import transforms
import PIL

# Define epsilon value for logit-Laplace distribution
logit_laplance_epsilon: float = 0.1

# The decoder of the dVAE produces six feature maps representing the sufficient statistics of the logit-Laplace distribution
def map_image(x: torch.Tensor) -> torch.Tensor:
    # Before feeding an image into the dVAE encoder, we transform its values using φ : [0, 255] → (ε, 1 − ε)
    # The equation to do it is the following --> x → (1−2ε)x + ε
    return (1 - 2 * logit_laplance_epsilon) * x + logit_laplance_epsilon


# To reconstruct an image for manual inspection or computing metrics, we compute xˆ = φ−1 (sigmoid(μ)), where μ is given by the first three feature maps output by the dVAE decoder
def unmap_image(x: torch.Tensor) -> torch.Tensor:
    # The equation to do it is the following --> (x - ε) / (1 - 2*ε)
    return torch.clamp(((x - logit_laplance_epsilon) / (1 - 2 * logit_laplance_epsilon)), min=0, max=1)  # Clamps all elements in input into the range [ min, max ]


# Data augmentation processing for images
def preprocess_image(img: typing.Union[torch.Tensor, PIL.Image.Image], target_img_size: int) -> torch.Tensor:
    # Get the minimum size of input img
    s = min(img.size()[1], img.size()[2])

    if s < target_img_size:
        raise ValueError(f'Minimum dimension for image {s} is {target_img_size}')
    
    # Get size
    size = (
        round(target_img_size / s * img.size()[1]),
        round(target_img_size / s * img.size()[2])
    )

    # Check the type of input image
    if isinstance(img, torch.Tensor):
        img = transforms.functional.resize(img, size)
        img = transforms.functional.center_crop(img, output_size=(2 * [target_img_size]))
        img = torch.unsqueeze(img, 0)
    elif isinstance(img, PIL.Image.Image):
        img = transforms.functional.resize(img, size, interpolation=PIL.Image.LANCZOS)
        img = transforms.functional.center_crop(img, output_size=(2 * [target_img_size]))
        img = torch.unsqueeze(transforms.ToTensor()(img), 0)
    else:
        raise TypeError(f'Input image can only be of either type torch.Tensor or PIL.Image.Image, receive {type(img)}')

    return  map_image(img)

 
# Modified version of nn.Conv2d from https://github.com/openai/DALL-E/blob/master/dall_e/utils.py
class ModifiedConv2d(nn.Module):
    def __init__(
        self,
        in_planes: int,    # Input channels
        out_planes: int,   # Output channels
        kernel_width: int  # Basically kernel size used in nn.Conv2d
    ):
        super().__init__()

        # Define weight
        weight = torch.empty(size=(out_planes, in_planes, kernel_width, kernel_width))  # Tensor filled with uninitialized data of the shape passed in "size"
        weight.normal_(std=(1 / math.sqrt(in_planes * kernel_width ** 2)))  # Fills "weight" with elements samples from the normal distribution parameterized by std

        # Define bias
        bias = torch.zeros(size=(out_planes,))

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.padding = (kernel_width - 1) // 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply2D convolution over the input x
        return torch.nn.functional.conv2d(input=x, weight=self.weight, bias=self.bias, padding=self.padding)
