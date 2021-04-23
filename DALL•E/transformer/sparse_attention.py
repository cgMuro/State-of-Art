import torch
import torch.nn as nn
import numpy as np
import einops


class Attention(nn.Module):
    ''' A faster implementation of normal attention (the upper triangle is not computed, and many operations are fused) '''
    def __init__(
        self, 
        n_embeddings: int,              # Embedding dimension of the input
        n_heads: int,                   # Number of heads
        head_dim: int = 64,             # Number of dimensions for each head
        attention_mode: str = 'normal', # Type of attention (normal, strided, fixed)
        dropout: float = 0.5            # Dropout value
    ):
        super().__init__()

        self.n_heads = n_heads
        self.attention_mode = attention_mode
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
        b, n, c = x.shape

        # Calculate query, key and value vectors
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # Reshape and decompose qkv to get query, key and value vectors individually
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)

        # Calculate the scores and normalize (dividing by the square root of head_dim)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Apply mask if required
        if self.attention_mode:
            #  Get mask
            mask = get_attention_mask(n=dots.size()[2], attention_mode=self.attention_mode)
            # Rearrange mask
            mask = einops.rearrange(mask, 'b j -> b () () j')
            # Fill the scores (the "dots" matrix) with the mask values
            dots.masked_fill_(mask == 1, float('-inf'))
            del mask

        # Softmax of the scores
        attention = dots.softmax(dim=-1)

        # Multiply the value vectors to the corresponding scores
        out = torch.einsum('b h i j, b h j d -> b h i d', attention, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        # Project the output vector (if needed)
        out = self.to_out(out)

        return out


def get_attention_mask(n: int, attention_mode: str, local_attention_ctx: int = 32):
    ''' Generate 3 types of mask: normal, local, strided. Based on https://github.com/openai/sparse_attention/blob/c53f3bdbf6225be0582f0357072e82b13c69be7d/attention.py '''
    if attention_mode == 'normal':
        b = torch.tril(torch.ones((n, n)), diagonal=0)
    elif attention_mode == 'local':
        bandwith = local_attention_ctx
        ctx = min(n - 1, bandwith - 1)
        
        if ctx < 0:
            b = torch.tril(torch.ones((n, n)), diagonal=0)
        else:
            b = torch.tril(torch.ones((n, n)), diagonal=0) - torch.triu(torch.ones((n, n)), diagonal=-ctx)
            b.masked_fill_(b == 1, 2)
            b.masked_fill_(b == 0, 1)
            b.masked_fill_(b == -1, 0)
            b.masked_fill_(b == 2, 0)

    elif attention_mode == 'strided':
        stride = local_attention_ctx
        x = torch.arange(n, dtype=torch.int32).view(n, 1)
        y = torch.transpose(x, 0, 1)
        z = torch.zeros([n, n], dtype=torch.int32)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = torch.tensor((q - k) % stride) == torch.tensor(0, dtype=torch.int)
        c3 = torch.logical_and(c1, c2)
        b = c3.type(torch.float32)
    else:
        raise ValueError('Not yet implemented')
    b = b.view([1, 1, n, n])

    return b.type(torch.int)
