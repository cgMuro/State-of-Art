# https://github.com/lucidrains/vit-pytorch

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    """ Define Feed Forward Network with GELU activation and dropout """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """ Define behaviour for the Multi-headed attention block with masking """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)  # Check if we need to project the last vector

        self.heads = heads
        self.scale = dim_head ** -0.5

        # Define network to calculate query, value and key vectors
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Define network to project the last vector, otherwise use the identity matrix
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        # Get input shape and heads number
        b, n, c = x.shape()
        h = self.heads
        # Calculate query, key and value vectors
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # Reshape and decompose the qkv to get the query, key and value vectors individually
        q, k, v = map(lambda t: t.view(b, n, h, c // h).transpose(1, 2), qkv) # rearrange(t, 'b n (h d) -> b h n d', h=h)
        # Calculate the scores and normalize (dividing by the square root of dim_heads)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Calculate the mask value (used to reduce the importance of the masked vectors)
        mask_value = -torch.finfo(dots.dtype).max
        # Apply mask if required
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            # Check for errors
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'

            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill(~mask, mask_value)
            del mask

        # Softmax of the scores
        attn = dots.softmax(dim=-1)

        # Multiply the value vectors to the corresponding scores
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.traspose(1, 2).view(b, n, h*c)  # rearrange(out, 'b h n d -> b n (h d)')
        # Project the output vector (if needed)
        out = self.to_out(out)

        return out

class Trasformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        # Define list of modules
        self.layers = nn.ModuleList([])
        # Add module to the list based on the depth passed
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # Multi-headed attention block
                nn.LayerNorm(dim),                                                      
                Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                # Feed forward block
                nn.LayerNorm(dim),
                FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout)
            ]))

    def forward(self, x, mask=None):
        # Iterate through the list of modules
        for attn, ff in self.layers:
            x = x + attn(x, mask=mask)  # Residual connection + attention
            x = x + ff(x)               # Residual connection + feed forward
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        # Check for errors
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size'
        assert pool in { 'cls', 'mean' }, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Get the total number of patches
        num_patches = (image_size // patch_size) ** 2
        # Get dimension of each patch
        patch_dim = channels * patch_size ** 2

        # Define the network that will handle the patches
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim)
        )
        # Define positional and token embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # Define dropout
        self.dropout = nn.Dropout(emb_dropout)
        # Define transformer
        self.transformer = Trasformer(dim=dim, depth=depth, heads=heads,  dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
        
        self.pool = pool
        self.to_latent = nn.Identity()  # Identity matrix

        # Define the last layer that will handle the classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, mask=None):
        # Patch image and get dimensions
        x = self.to_patch_embedding(img)
        b, n, c = x.shape

        # Tokens (+ dropout)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n+1)]
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x, mask)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # If mean -> take a mean of the output in the dim=1. If cls -> take only the CLS token

        # Classification
        x = self.to_latent(x)
        x = self.mlp_head(x)

        return x
