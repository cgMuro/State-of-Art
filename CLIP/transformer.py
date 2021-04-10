import torch
import torch.nn as nn
import einops
from utils import ModifiedLayerNorm

class Transformer(nn.Module):
    """ Architecture and logic for the transformer """
    def __init__(
        self, 
        width: int, 
        n_blocks: int, 
        n_heads: int = 8, 
        head_dim: int = 64, 
        dropout: float = 0.5,
        mask: torch.Tensor = None
    ):
        super().__init__()

        # Define list of modules
        self.layers = nn.ModuleList([])
        # Add module to the list based on the n_blocks passed
        for _ in range(n_blocks):
            self.layers.append(nn.ModuleList([
                # Multi-headed attention block
                Attention(width=width, n_heads=n_heads, head_dim=head_dim, dropout=dropout, mask=mask),
                # Feed Forward Block
                FeedForward(width=width, dropout=dropout)
            ]))

        # Define modified layer normalization
        self.layernorm_1 = ModifiedLayerNorm(width)
        self.layernorm_2 = ModifiedLayerNorm(width)

    def forward(self, x: torch.Tensor, mask=None):
        # Iterate through the list of modules
        for attention, feedforward in self.layers:
            x = x + attention(self.layernorm_1(x), mask=mask)  # Residual connection + attention
            x = x + feedforward(self.layernorm_2(x))          # Residual connection + feed forward
        return x

class Attention(nn.Module):
    """ Defines behaviour for the Multi-headed attention block with masking """
    def __init__(
        self, 
        width: int, 
        n_heads: int = 8, 
        head_dim: int = 64, 
        dropout: float = 0.5,
        mask: torch.Tensor = None
    ):
        super().__init__()

        self.n_heads = n_heads
        self.scale = width ** -0.5

        inner_dim = head_dim * n_heads
        project_out = not (n_heads == 1 and head_dim == width)   # Check if we need to project the last vector

        # Define network to calculate query, value and key vectors
        self.to_qkv = nn.Linear(width, inner_dim * 3, bias=False)

        # Define network to project the last vector, otherwise use the identity matrix
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, width),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor):
        # Get input shape
        b, n, c = x.shape()

        # Calculate query, key and value vectors
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # Reshape and decompose qkv to get query, key and value vectors individually
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads))
        # Calculate the scores and normalize (dividing by the square root of head_dim)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Calculate the mask value (used to reduce the importance of the masked vectors)
        mask_value = -torch.finfo(dots.dtype).max

        # Apply mask if required
        if mask is not None:
            mask = nn.functional.pad(mask.flatten(1), (1, 0), value=True)
            # Check for errors
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'

            mask = einops.rearrange(mask, 'b i -> b () i ()') * einops.rearrange(mask, 'b j -> b () () j')
            dots.masked_fill(~mask, mask_value)
            del mask

        # Softmax of the scores
        attention = dots.softmax(dim=-1)

        # Multiply the value vectors to the corresponding scors
        out = torch.einsum('b h i j, b h j d -> b h i d', attention, v)
        out = einops.rearrange(out, 'b h m d -> b n (h d)')
        # Project the output vector (if needed)
        out = self.to_out(out)

        return out

class FeedForward(nn.Module):
    """ Defines Feed Forward Network with GELU activation and dropout """
    def __init__(
        self, 
        width: int,
        dropout: float = 0.5
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
        