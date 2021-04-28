import math
import torch
import torch.nn as nn
import einops
from dVAE.dvae import dVAE
from dVAE.utils import unmap_image
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

        # Output projection network
        total_tokens = 32*32 + max_length   # image tokens + max text tokens

        self.to_logits = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, total_tokens)
        )

    @torch.no_grad()
    def generate_images(self, text: torch.Tensor, image: torch.Tensor = None):
        # Make sure text is of the right length
        text = text[:, :self.max_len]

        if image is not None:
            # Get image embedding
            image_embedding = self.dvae.encoder(image)
            image_embedding = nn.functional.gumbel_softmax(image_embedding, tau=1.0, hard=False, dim=1)
            # Get 14*32 tokens for priming
            # image_embedding = image_embedding[:, :, :14, :14]
        else:
            # Pass in empty array to represent image
            image_embedding = text[:, self.max_len:]

        # Transformer
        logits = self.transformer(text, image_embedding)
        logits = self.to_logits(logits)
        # Get only the image from logits then rearrange
        logits = logits[self.max_len:, :, self.max_len:]
        logits = einops.rearrange(logits, 'c b (w h) -> b c w h', h=int(math.sqrt(logits.size()[-1])))  # channels batch_size tokens ->  batch_size channels width height

        # Get actual image
        images = self.dvae.decoder(logits)

        return unmap_image(images)

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        # dVAE -> get image embedding
        image_embedding, image_logits = self.dvae(image)

        # Transformer -> get tokens
        tokens = self.transformer(text, image_embedding)
        logits = self.to_logits(tokens)

        # Process image embedding
        image_embedding = torch.argmax(image_embedding, dim=1)
        image_embedding = einops.rearrange(image_embedding, 'b h w -> b (h w)')

        return image_embedding, logits
