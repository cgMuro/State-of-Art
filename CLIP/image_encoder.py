import torch
import torch.nn as nn
from transformer import Transformer
from utils import ModifiedLayerNorm

class ViT(nn.Module):
    """ Architecture and logic for the image encoder, which is a visual transformer """
    def __init__(
        self,
        image_size: int,           # Size of the image to process
        patch_size: int,           # Size of the patch in which the input image will be divided
        width: int,                # Dimension for the embeddings 
        n_blocks: int,             # Number of blocks that compose the transformer (i.e. the depth)
        output_dim: int,           # Dimension of the output
        n_heads: int = 8,          # Number of heads for each multi-head attention layer
        channels: int = 3,         # Number of image's channels
        head_dim: int = 64,        # Dimension of each multi-head layer
        mask: torch.Tensor = None, # Define mask to be used in multi-head attention
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
        self.transformer = Transformer(width=width, n_blocks=n_blocks, n_heads=n_heads, head_dim=head_dim, dropout=dropout, mask=mask)

        # Define output projection layer
        self.out_proj = nn.Parameter(scale * torch.rand(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)   # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)   # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.layernorm_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        x = self.layernorm_post(x[:, 0, :])

        x = x @ self.out_proj

        return x
