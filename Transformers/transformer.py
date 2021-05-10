import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """ Define the full Transformer behaviour """
    def __init__(self, vocab_size: int, n_embeddings: int, n_heads: int, n_layers: int, max_len: int, dropout: float = 0.5):
        super(Transformer, self).__init__()
        # Define dropout
        self.dropout = nn.Dropout(dropout)
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, n_embeddings)
        # Positional embedding
        self.positional_embedding = PositionalEncoding(vocab_size=vocab_size, n_embeddings=n_embeddings, max_len=max_len, dropout=dropout)
        # Encoder Block
        self.encoder = EncoderBlock(n_embeddings=n_embeddings, n_heads=n_heads, n_layers=n_layers, max_len=max_len, dropout=dropout)
        # Decoder Block
        self.decoder = DecoderBlock(n_embeddings=n_embeddings, n_heads=n_heads, n_layers=n_layers, max_len=max_len, dropout=dropout)
        # Final Network
        self.fn = nn.Linear(n_embeddings, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embeddings
        token_embeddings = self.token_embedding(x)
        positional_embeddings = self.positional_embedding(token_embeddings)
        x = self.dropout(token_embeddings + positional_embeddings)

        # Encoder
        encoder_out = self.encoder(x)

        # Decoder
        decoder_out = self.decoder(x, encoder_out)

        # Output network
        res = self.dropout(self.fn(decoder_out))
        res = F.softmax(res, dim=-1)

        return res

class PositionalEncoding(nn.Module):
    """ Define the process to calculate the positional embeddings """
    def __init__(self, vocab_size: int, n_embeddings: int, max_len: int, dropout: float = 0.5):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, n_embeddings)
        position = torch.arange(start=0, end=max_len, dtype=torch.float).unsqueeze(1)

        # Sinusoidal encoding
        pe[:, 0::2] = torch.sin(
            position * torch.exp(torch.arange(0, n_embeddings, step=2).float() * (-math.log(10000.0) / n_embeddings))
        )
        pe[:, 1::2] = torch.cos(
            position * torch.exp(torch.arange(0, n_embeddings, step=2).float() * (-math.log(10000.0) / n_embeddings))
        )

        pe = pe.unsqueeze(0).transpose(0, 1)

        # Add a buffer to the module
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.pe[:x.size(0), :])
        return x

class EncoderBlock(nn.Module):
    def __init__(self, n_embeddings: int, n_heads: int, n_layers: int, max_len: int, dropout: float = 0.5):
        super(EncoderBlock, self).__init__()

        # Define Multi-headed attention
        self.multiheadattention = MultiHeadedAttention(n_embeddings=n_embeddings, n_heads=n_heads, n_layers=n_layers, max_len=max_len, dropout=dropout)
        # Define Feed Forward Network
        self.fcn = nn.Sequential(
            nn.Linear(n_embeddings, 4*n_embeddings),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(4*n_embeddings, n_embeddings),
            nn.Dropout(dropout),
        )
        # Define normalization
        self.norm = nn.LayerNorm(n_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.norm(self.multiheadattention(x))
        x = x + self.norm(self.fcn(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, n_embeddings: int, n_heads: int, n_layers: int, max_len: int, dropout: float = 0.5):
        super(DecoderBlock, self).__init__()

        # Define Multi-Headed attention
        self.multiheadattention = MultiHeadedAttention(n_embeddings=n_embeddings, n_heads=n_heads, n_layers=n_layers, max_len=max_len, dropout=dropout)
        # Define Feed Forward Network
        self.fcn = nn.Sequential(
            nn.Linear(n_embeddings, 4*n_embeddings),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(4*n_embeddings, n_embeddings),
            nn.Dropout(dropout),
        )
        # Define normalization
        self.norm = nn.LayerNorm(n_embeddings)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        x = x + self.norm(self.multiheadattention(x, mask=True))
        x = x + self.norm(self.multiheadattention(x, encoder_out, mask=True))
        x = x + self.norm(self.fcn(x))
        return x

class MultiHeadedAttention(nn.Module):
    """ Define the Multi-headed Attention Block with masking """
    def __init__(self, n_embeddings: int, n_heads: int, n_layers: int, max_len: int, dropout: float = 0.5):
        super(MultiHeadedAttention, self).__init__()

        self.n_heads = n_heads
        self.n_embeddings = n_embeddings

        # Matrices for query, key and value vectors
        self.query = nn.Linear(n_embeddings, n_embeddings)
        self.key = nn.Linear(n_embeddings, n_embeddings)
        self.value = nn.Linear(n_embeddings, n_embeddings)
        # Projection for vector z
        self.projection = nn.Linear(n_embeddings, n_embeddings)

        # Normalization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_embeddings)

        # Masking
        self.register_buffer(
            'mask', 
            torch.tril(torch.ones(size=(max_len, max_len))).view(1, 1, max_len, max_len)
        )

    def forward(self, x: torch.Tensor, encoder_out=None, mask=None) -> torch.Tensor:
        B, T, C = x.size()
        # Get key, query and value vectors
        if encoder_out != None:
            k = self.key(encoder_out).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)    # Extract key vector from encoder's output
            q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)            # Extract query vector from input vector
            v = self.value(encoder_out).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # Extract value vector from encoder's output
        else:
            k = self.key(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
            q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
            v = self.value(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        # Calculate and normalize score
        score = (q @ k.transpose(-1, -2)) / math.sqrt(self.n_embeddings)
        # Mask if requested
        if mask:
            score = score.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        # Use softmax
        score = F.softmax(score, dim=-1)
        # Get final z vector
        z = (score @ v).transpose(1, 2).contiguous().view(B, T, C) 
        # Project z vector (best way to combine it) and apply dropout
        z = self.dropout(self.projection(z))

        return z
