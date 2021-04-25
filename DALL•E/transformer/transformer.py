import torch
import torch.nn as nn
import einops
import functools
import typing
from .sparse_attention import Attention


class Transformer(nn.Module):
    """ Architecture and logic for transformer """
    def __init__(
        self,
        output_dim: int,                # Dimension of the output
        width: int,                     # Dimension for the embeddings
        n_blocks: int,                  # Number of blocks the transformer will have (i.e. its depth)
        n_heads: int,                   # Number of heads for the multihead attention
        head_dim: int = 64,             # Number of dimensions for each head in multihead attention
        max_length: int = 256,          # Define maximum sequence length
        vocab_size: int = 16384,        # Define size of vocabulary
        dropout: float = 0.5,           # Dropout value
    ):
        super().__init__()

        # Define token and positional embeddings
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=width)
        self.positional_embedding = nn.Parameter(torch.empty(max_length, width))
 
        # Define layer normalization
        self.layernorm = nn.LayerNorm(width)

        # Define list of modules
        self.layers = nn.ModuleList([])

        # Add module to the list based on the "n_blocks" passed
        for idx in range(n_blocks):
            # Define the type of attention mask
            if (idx-2) // 4 == 0:
                # attention_mode = 'column'
                attention_mode = 'normal'
            elif idx == n_blocks:
                # attention_mode = 'convolutional'
                attention_mode = 'normal'
            else:
                attention_mode = 'row'
            
            # Build block
            self.layers.append(nn.ModuleList([
                # Multi-head attention block
                LayerScale(width, idx + 1, Attention(n_embeddings=width, n_heads=n_heads, head_dim=head_dim, attention_mode=attention_mode, dropout=dropout)),
                # Feedforward block
                LayerScale(width, idx + 1, FeedForward(width=width, dropout=dropout))
            ]))

        # Define layer normalization
        self.layernorm_1 = nn.LayerNorm(width)
        self.layernorm_2 = nn.LayerNorm(width)

        # Define projection
        self.projection = nn.Parameter(torch.empty(width, output_dim))

    def forward(self, text: torch.Tensor, image_embedding: torch.Tensor) -> torch.Tensor:
        # Get index of the highest number along last dimension
        # idx_max_n = text.argmax(dim=-1)

        # # Add padding to text -> used to differentiate between text tokens and image embedding
        # text = nn.functional.pad(text, (0, 3), value=0)

        # Apply token and positional embeddings
        text = self.token_embedding(text)        # shape = [batch_size, n_ctx, d_model]
        text = text + self.positional_embedding
        
        # Rearrange image embedding
        # text = text.permute(1, 0, 2)
        image_embedding = einops.rearrange(image_embedding, 'b f w h -> b f (w h)')

        # Concatenate text embedding and image embedding
        x = torch.cat((text, image_embedding), dim=1)

        # Iterate through the list of modules and pass the input
        for attention, feedforward in self.layers:
            x = x + attention(self.layernorm_1(x))     # Residual connection + attention
            x = x + feedforward(self.layernorm_2(x))   # Residual connection + feedforward

        # Permute and normalize
        x = x.permute(1, 0, 2)  # shape = [batch_size, n_ctx, transformer.width]
        x = self.layernorm(x)

        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), idx_max_n] @ self.projection
        x = x @ self.projection

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
