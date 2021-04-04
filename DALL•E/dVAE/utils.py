import random
import torch
from torchvision import transforms
import tensorflow as tf

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
def preprocess_image(img: torch.Tensor, target_res: int = 256, channel_count: int = 3) -> torch.Tensor:
    h, w = img.size()[0], img.size()[1]
    s_min = torch.min(torch.tensor(h), torch.tensor(w))

    # Apply random crop
    img = image_random_crop(img, output_size=(2 * [s_min] + [3]))

    # Get random uniform distribution
    t_min = torch.min(s_min, torch.tensor(round(9 / 8 * target_res)))
    t_max = torch.min(s_min, torch.tensor(round(12 / 8 * target_res)))
    t = (t_min - (t_max + 1)) * torch.rand([]) + (t_max + 1)

    # Resize image -> I had to use tensorflow here because of the lack of alternatives in PyTorch (torchvision.transforms.functional.resize apparently only works with PIL images for now)
    t = tf.convert_to_tensor(t.numpy())   # Convert torch.Tensor to numpy array and then to tf.Tensor
    img = tf.image.resize(img, size=[t, t], method=tf.image.ResizeMethod.AREA)  # Apply image resize with area interpolation
    img = torch.from_numpy(img.numpy())   # Convert tf.Tensor to numpy array and then back to torch.Tensor 

    # Clamp the image values between 0 and 255, round the resulting tensor and change the type to uint8
    img = (torch.round(torch.clamp(img, min=0, max=255))).type(torch.uint8)

    # Apply random crop
    img = image_random_crop(img, output_size=(2 * [target_res] + [channel_count]))

    # Randomly flip the image orizontally, with a 1 in 2 chance
    if random.random() < 0.5:
        return transforms.functional.hflip(img)
    else:
        return img
        
# Function that applies random crop
def image_random_crop(img, output_size):
    h, w = img.size()[0], img.size()[1]
    th, tw, _ = output_size
    if w == tw and h == th:
        return transforms.functional.crop(img, 0, 0, h, w)
    else:
        return transforms.functional.crop(img, random.randint(0, h - th), random.randint(0, w - tw), h, w)
