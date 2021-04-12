import numpy as np
import torch
import torch.nn as nn
from image_encoder import ViT
from text_encoder import TransformerTextEncoder

class CLIP(nn.Module):
    def __init__(
        self,
        emb_dim: int = 512,        # Multi-modal embedding space of images and text encodings
        # Parameters for images
        image_size: int = 224,     # Size of the input image
        patch_size: int = 32,      # Size of the patch in which the image will be divided
        vision_blocks: int = 12,   # Number of blocks that compose the vision transformer (i.e. the depth) [possible values from the paper: 12, 24]
        vision_width: int = 768,   # Dimension for the image embeddings [possible values from the paper: 768, 1024]
        vision_heads: int = 12,    # Number of heads for each multihead attention layer in the vision transformer [possible values from the paper: 12, 16]
        # Parameters for text
        max_length: int = 76,      # Maximum sequence length
        vocab_size: int = 49408,   # Size of the text's vocabulary
        text_blocks: int = 12,     # Number of blocks that compose the transformer (i.e. the depth) [possible values from the paper: 12, 16]
        text_width: int = 512,     # Dimension for the text embeddings [possible values from the paper: 512, 768]
        text_heads: int =  8       # Number of heads for each multi-head attention layer in the transformer [possible values from the paper: 8, 12]
    ):
        super().__init__()

        # Define image encoder (Vision Transformer)
        self.image_encoder = ViT(image_size=image_size, patch_size=patch_size, output_dim=emb_dim, width=vision_width, n_blocks=vision_blocks, n_heads=vision_heads, channels=3, head_dim=64, mask=None, dropout=0.5)

        # Define text encoder (vanilla Transformer)
        self.text_encoder = TransformerTextEncoder(output_dim=emb_dim, vocab_size=vocab_size, max_length=max_length, width=text_width, n_blocks=text_blocks, n_heads=text_heads, head_dim=64, dropout=0.5, tensor_type=self.image_encoder.conv1.weight.dtype)

        # Define logit scale -> scales pairwise cosine similarities
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # We use numpy because it seems to be more precise

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        # Get image and text features
        image_features = self.image_encoder(image.type(self.image_encoder.conv1.weight.dtype))
        text_features = self.text_encoder(text)

        # Normalize features
        image_embedding = image_features / image_features.norm(dim=-1, keepdim=True)
        text_embedding = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity as logits
        logits_per_image = self.logit_scale.exp() * image_embedding @ text_embedding.t()
        logits_per_text = self.logit_scale.exp() * text_embedding @ image_embedding.t()

        return logits_per_image, logits_per_text
