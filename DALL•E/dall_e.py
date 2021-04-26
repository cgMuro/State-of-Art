import math
import torch
import torch.nn as nn
import einops
from dVAE.dvae import dVAE
from transformer.transformer import Transformer

class DALLE(nn.Module):
    def __init__(
        self,
        # dVAE
        in_planes: int = 3,
        hidden_planes: int = 64,
        out_planes: int = 3,
        blocks_per_group: int = 1,
        dVAE_vocab_size: int = 8192,
        # Transformer
        output_dim: int = 32,
        # width: int = 64,
        n_block: int = 1,
        n_heads: int = 1,
        head_dim: int = 1,
        max_length: int = 256,
        transformer_vocab_size: int = 16384,
        dropout: float = 0.5
    ):
        super().__init__()

        self.max_len = max_length
        
        # dVAE
        self.dvae = dVAE(in_planes=in_planes, hidden_planes=hidden_planes, out_planes=out_planes, blocks_per_group=blocks_per_group, vocab_size=dVAE_vocab_size)

        # Transformer
        # width = image width * image height
        width = 1024
        self.transformer = Transformer(output_dim=output_dim, width=width, n_blocks=n_block, n_heads=n_heads, head_dim=head_dim, max_length=max_length, vocab_size=transformer_vocab_size, dropout=dropout)

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        # dVAE -> get image embedding
        image_embedding, _ = self.dvae(image)
        # Transformer -> get tokens
        tokens = self.transformer(text, image_embedding)

        return tokens
