import torch
import torch.nn as nn
from transformer import Transformer
from utils import ModifiedLayerNorm

class ViT(nn.Module):
    """ Architecture and logic for the image encoder, which is a visual transformer """
    def __init__(
        self,
        image_size: int,           # Size of the image to process [possible values from the paper: 224, 336]
        patch_size: int,           # Size of the patch in which the input image will be divided
        output_dim: int,           # Dimension of the output [possible values from the paper: 512, 768]
        width: int = 768,          # Dimension for the image embeddings [possible values from the paper: 768, 1024]
        n_blocks: int = 12,        # Number of blocks that compose the transformer (i.e. the depth) [possible values from the paper: 12, 24]
        n_heads: int = 12,         # Number of heads for each multihead attention layer [possible values from the paper: 12, 16]
        channels: int = 3,         # Number of image's channels
        head_dim: int = 64,        # Dimension of each multihead layer
        mask: torch.Tensor = None, # Define mask to be used in multihead attention
        dropout: float = 0.5       # Define dropout
    ):
        super().__init__()

        # Define scaling value
        scale = width ** -0.5

        # Define convolutional layer -> it's used to patch the image
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # Define positional and token embeddings
        self.class_embedding = nn.Parameter(scale * torch.rand(width))
        self.positional_embedding = nn.Parameter(scale * torch.rand((image_size // patch_size) ** 2 + 1, width))
        
        # Define modified layer normalizations
        self.layernorm_pre = ModifiedLayerNorm(width)
        self.layernorm_post = ModifiedLayerNorm(width)

        # Define transformer
        self.transformer = Transformer(n_embeddings=width, n_blocks=n_blocks, n_heads=n_heads, head_dim=head_dim, dropout=dropout)

        # Define output projection layer
        self.out_proj = nn.Parameter(scale * torch.rand(width, output_dim))

    def forward(self, x: torch.Tensor):
        # Patch the input
        x = self.conv1(x)   # shape = [*, width, grid, grid]
        # Reshape and permute
        x = x.reshape(x.shape[0], x.shape[1], -1)   # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # Get embeddings and normalize
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.layernorm_pre(x)
        # Permute and pass the input into the transformer, then permute again
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        # Normalize
        x = self.layernorm_post(x[:, 0, :])
        # Apply output projection
        x = x @ self.out_proj

        return x
