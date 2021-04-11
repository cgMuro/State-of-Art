import torch
import torch.nn as nn
import einops
from utils import ModifiedLayerNorm

class Transformer(nn.Module):
    """ Architecture and logic for transformer """
    def __init__(
        self, 
        n_embeddings: int,           # Embedding dimension of the input
        n_blocks: int,               # Number of blocks the transformer will have (i.e. its depth)
        n_heads: int,                # Number of heads for the multihead attention
        head_dim: int = 64,          # Number of dimensions for each head in multihead attention
        dropout: float = 0.5,        # Dropout value
        mask: torch.Tensor = None    # Mask value
    ):
        super().__init__()

        self.mask = mask

        # Define list of modules
        self.layers = nn.ModuleList([])
        # Add module to the list based on the "n_blocks" passed
        for _ in range(n_blocks):
            self.layers.append(nn.ModuleList([
                # Multi-head attention block
                Attention(n_embeddings=n_embeddings, n_heads=n_heads, head_dim=head_dim, dropout=dropout, mask=mask),
                # Feedforward block
                FeedForward(width=n_embeddings, dropout=dropout)
            ]))

        # Define modified layer normalization
        self.layernorm_1 = ModifiedLayerNorm(n_embeddings)
        self.layernorm_2 = ModifiedLayerNorm(n_embeddings)

    def forward(self, x: torch.Tensor):
        # Iterate through the list of modules and pass the input
        for attention, feedforward in self.layers:
            x = x + attention(self.layernorm_1(x), mask=self.mask)  # Residual connection + attention
            x = x + feedforward(self.layernorm_2(x))                # Residual connection + feedforward
        return x

class Attention(nn.Module):
    """ Defines behaviour for the Multi-headed attention block with masking """
    def __init__(
        self, 
        n_embeddings: int,          # Embedding dimension of the input
        n_heads: int,               # Number of heads
        head_dim: int = 64,         # Number of dimensions for each head
        dropout: float = 0.5,       # Dropout value
        mask: torch.Tensor = None   # Mask value
    ):
        super().__init__()

        self.n_heads = n_heads
        self.mask = mask
        self.scale = n_embeddings ** -0.5
        inner_dim = head_dim * n_heads
        project_out = not (n_heads == 1 and head_dim == n_embeddings)   # Check if we need to project the last vector

        # Define network to calculate query, value and key vectors
        self.to_qkv = nn.Linear(n_embeddings, inner_dim * 3, bias=False)

        # Define network to project the last vector, otherwise use the identity matrix
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, n_embeddings),
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
        if self.mask is not None:
            # Pad mask
            mask = nn.functional.pad(self.mask.flatten(1), (1, 0), value=True)
            # Check for errors
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'
            # Rearrange mask
            mask = einops.rearrange(mask, 'b i -> b () i ()') * einops.rearrange(mask, 'b j -> b () () j')
            # Fill the scores (the "dots" matrix) with the mask values
            dots.masked_fill(~mask, mask_value)
            # Delete the mask
            del mask

        # Softmax of the scores
        attention = dots.softmax(dim=-1)

        # Multiply the value vectors to the corresponding scores
        out = torch.einsum('b h i j, b h j d -> b h i d', attention, v)
        out = einops.rearrange(out, 'b h m d -> b n (h d)')
        # Project the output vector (if needed)
        out = self.to_out(out)

        return out

class FeedForward(nn.Module):
    """ Defines Feedforward Network with GELU activation and dropout """
    def __init__(
        self, 
        width: int,             # Layer's width
        dropout: float = 0.5    # Dropout value
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
