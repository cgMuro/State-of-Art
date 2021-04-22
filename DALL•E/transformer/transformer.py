import torch
import torch.nn as nn
import einops
import functools
import typing
from sparse_attention import Attention


class Transformer(nn.Module):
    """ Architecture and logic for transformer """
    def __init__(
        self, 
        emb_dim: int,                   # Embedding dimension of the input
        n_blocks: int,                  # Number of blocks the transformer will have (i.e. its depth)
        n_heads: int,                   # Number of heads for the multihead attention
        head_dim: int = 64,             # Number of dimensions for each head in multihead attention
        attention_mode: str = 'normal', # Type of attention mask to implement
        dropout: float = 0.5,           # Dropout value
    ):
        super().__init__()
 
        # Define layer normalization
        self.layernorm = nn.LayerNorm(emb_dim)

        # Define list of modules
        self.layers = nn.ModuleList([])

        # Add module to the list based on the "n_blocks" passed
        for idx in range(n_blocks):
            # Define the type of attention mask
            if (idx-2) // 4 == 0:
                attention_mode = 'column'
            elif idx == len(n_blocks)-1:
                attention_mode = 'convolutional'
            else:
                attention_mode = 'row'
            
            # Build block
            self.layers.append(nn.ModuleList([
                # Multi-head attention block
                LayerScale(emb_dim, idx + 1, self.layernorm(Attention(n_embeddings=emb_dim, n_heads=n_heads, head_dim=head_dim, attention_mode=attention_mode, dropout=dropout))),
                # Feedforward block
                LayerScale(emb_dim, idx + 1, self.layernorm(FeedForward(width=emb_dim, dropout=dropout)))
            ]))

        # Define layer normalization
        self.layernorm_1 = nn.LayerNorm(emb_dim)
        self.layernorm_2 = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor):
        # Iterate through the list of modules and pass the input
        for attention, feedforward in self.layers:
            x = x + attention(self.layernorm_1(x))     # Residual connection + attention
            x = x + feedforward(self.layernorm_2(x))   # Residual connection + feedforward
        return x

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


class LayerScale(nn.Module):
    def __init__(self, dim: int, depth: int, net: nn.Module):
        super().__init__()
        # Define initial epsilon
        if depth <= 18:
            init_eps = 0.1
        elif depth >  18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        # Define scale
        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)

        self.net = net

    def forward(self, x):
        return self.net(x) * self.scale        
