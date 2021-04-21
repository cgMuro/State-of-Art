import numpy as np
import torch
import torch.nn as nn
import PIL
from typing import Tuple, Union, List
from image_encoder import ViT
from text_encoder import TransformerTextEncoder
from utils import augment_image
from tokenizer import SimpleTokenizer

class CLIP(nn.Module):
    def __init__(
        self,
        emb_dim: int = 512,        # Multi-modal embedding space of images and text encodings
        temperature: Tuple[float, float] = (0.07, 100), # Define temperature values (initial, maximum)
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

        self.max_length = max_length
        self.image_size = image_size
        self.initial_temp = temperature[0]
        self.max_temp = temperature[1]

        # Define image encoder (Vision Transformer)
        self.image_encoder = ViT(image_size=image_size, patch_size=patch_size, output_dim=emb_dim, width=vision_width, n_blocks=vision_blocks, n_heads=vision_heads, channels=3, head_dim=64, dropout=0.5)

        # Define text encoder (vanilla Transformer)
        self.text_encoder = TransformerTextEncoder(output_dim=emb_dim, vocab_size=vocab_size, max_length=max_length, width=text_width, n_blocks=text_blocks, n_heads=text_heads, head_dim=64, dropout=0.5, tensor_type=self.image_encoder.conv1.weight.dtype)

        # Define logit scale -> scales pairwise cosine similarities
        self.logit_scale = nn.Parameter(torch.tensor([np.log(1 / self.initial_temp)]))  # We use numpy because it seems to be more precise

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        ''' Encodes an image '''
        return self.image_encoder(image.type(self.image_encoder.conv1.weight.dtype))

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        ''' Encodes text '''
        return self.text_encoder(text)
    
    def predict(
            self, 
            # model: torch.nn.Module, 
            images: Union[torch.Tensor, PIL.Image.Image], 
            texts: Union[torch.Tensor, List[str]], 
            tokenizer: SimpleTokenizer,
            device: torch.DeviceObjType, 
            top_k_returns: int = 5
        ):
            ''' Takes in a pretrained model, the processed images and texts, model's device, and returns the number ("top_k_returns") of top probabilities and labels '''

            # Process images
            images = torch.stack([augment_image(image, self.image_size) for image in images])
            # Tokenize texts if needed
            if isinstance(texts, list):
                text_input = torch.zeros(len(texts), self.max_length, dtype=torch.long)
                sot_token = tokenizer.encoder['<|startoftext|>']
                eot_token = tokenizer.encoder['<|endoftext|>']

                for i, tokens in enumerate(texts):
                    tokens = [sot_token] + tokens + [eot_token]
                    text_input[i, :len(tokens)] = torch.tensor(tokens)
                
                texts = text_input
            
            # Move tensors to device
            images = images.to(device)
            texts = texts.to(device)

            # Get prediction from model
            with torch.no_grad():
                # Get image features
                image_features = self.encode_image(images)
                # Get text features and normalize
                text_features = self.encode_text(texts)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Get text probabilities
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            # Get top 5 probabilities and labels
            top_probs, top_labels = text_probs.cpu().topk(top_k_returns, dim=-1)

            return top_probs, top_labels

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        # Get image and text features
        image_features = self.image_encoder(image.type(self.image_encoder.conv1.weight.dtype))
        text_features = self.text_encoder(text)

        # Normalize features
        image_embedding = image_features / image_features.norm(dim=-1, keepdim=True)
        text_embedding = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity as logits
        logit_scale = torch.clamp(self.logit_scale.exp(), max=self.max_temp).type(image_embedding.dtype)
        logits_per_image = logit_scale * image_embedding @ text_embedding.t()
        logits_per_text = logit_scale * text_embedding @ image_embedding.t()

        return logits_per_image, logits_per_text
