import numpy as np
import torch
import torch.nn as nn
from image_encoder import ViT
from text_encoder import TransformerTextEncoder

class CLIP(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        # Parameters for images
        image_size: int,
        patch_size: int,
        vision_blocks: int,
        vision_width: int,
        # Parameters for text
        max_length: int,
        vocab_size: int,
        text_blocks: int,
        text_width: int,
    ):
        super().__init__()

        # Define image encoder (Vision Transformer)
        self.image_encoder = ViT(image_size=image_size, patch_size=patch_size, width=vision_width, n_blocks=vision_blocks, output_dim=emb_dim, n_heads=8, channels=3, head_dim=64, mask=None, dropout=0.5)

        # Defina text encoder (Transformer)
        self.text_encoder = TransformerTextEncoder(vocab_size=vocab_size, max_length=max_length, width=text_width, n_blocks=text_blocks, output_dim=emb_dim, n_heads=8, head_dim=64, dropout=0.5)

        # Define logit scale -> scaled pairwise cosine similarities
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # We use numpy because it seems to be more precise

    def forward(self, image, text):
        # Get image and text features
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)

        # Normalize features
        image_embedding = image_features / image_features.norm(dim=-1, keepdim=True)
        text_embedding = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity as logits
        logits_per_image = self.logit_scale.exp() * image_embedding @ text_embedding.t()
        logits_per_text = self.logit_scale.exp() * text_embedding @ image_embedding.t()

        return logits_per_image, logits_per_text
