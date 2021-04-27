import torch
import typing
from tokenizer import SimpleTokenizer

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
