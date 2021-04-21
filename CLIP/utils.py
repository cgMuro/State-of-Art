import torch
import torch.nn as nn
from torchvision import transforms
import typing
import PIL
from tokenizer import SimpleTokenizer

class ModifiedLayerNorm(nn.LayerNorm):
    """ Subclass torch's LayerNorm to handle fp16 """
    def forward(self, x: torch.Tensor):
        orgin_type = x.dtype  # Store x's orginal type
        ret = super().forward(x.type(torch.float32))  # Apply nn.LayerNorm on x after changing its type to torch.float32
        return ret.type(orgin_type)  # Return x into its original type


# Image augmenter function
def augment_image(image: typing.Union[torch.Tensor, PIL.Image.Image], size: int):
    if isinstance(image, torch.Tensor):
        # Define transformations to apply for input image of type torch.Tensor
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
            transforms.RandomCrop(size),
            lambda image: image.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    elif isinstance(image, PIL.Image.Image):
        # Define transformations to apply for input image of type PIL.Image.Image
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
            transforms.RandomCrop(size),
            lambda image: image.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    else:
        raise TypeError(f'Image can be only of type either torch.Tensor or PIL.Image.Image. Type passed {type(image)}')

    # Apply transformations to passed image
    return transform(image)


# Applies tokenizer to input text -> returns the tokenized representation of the given input string(s)
def tokenize(tokenizer: SimpleTokenizer, text: typing.Union[str, typing.List[str]], max_length: int = 76):
    # If the input is a string transform it into an array
    if isinstance(text, str): text = [text]

    SOT_token = tokenizer.encoder["<|startoftext|>"]  # Encode Start Of Text token
    EOT_token = tokenizer.encoder["<|endoftext|>"]    # Encode End Of Text token
    all_tokens = [[SOT_token] + tokenizer.encoder(t) + [EOT_token] for t in text] # Encode all input text

    # Init 2-dimensional tensor that will contain all encoded tokens, shape=[number of input strings, max_length]
    result = torch.zeros(len(all_tokens), max_length, dtype=torch.long)

    # Add all tokens to "result"
    for idx, tokens in enumerate(all_tokens):
        # Check if token has length greater than the maximum length
        if len(tokens) > max_length:
            raise RuntimeError(f"Input {text[idx]} is too long for context length {max_length}")

        result[idx, :len(tokens)] = torch.tensor(tokens)

    return result
