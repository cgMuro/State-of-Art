import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """ Define the full Transformer behaviour """
    def __init__(self, vocab_size, n_embedding, n_heads, n_layers, max_len, dropout=0.5):
        super(Transformer, self).__init__()
        # Define dropout
        self.dropout = nn.Dropout(dropout)
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, n_embedding)
        # Positional embedding
        self.positional_embedding = PositionalEncoding(vocab_size=vocab_size, n_embeddings=n_embedding, max_len=max_len)
        # Encoder Block
        self.encoder = EncoderBlock(n_embedding=n_embedding, n_heads=n_heads, n_layers=n_layers, max_len=max_len, dropout=dropout)
        # Decoder Block
        self.decoder = DecoderBlock(n_embedding=n_embedding, n_heads=n_heads, n_layers=n_layers, max_len=max_len, dropout=dropout)
        # Final Network
        self.fn = nn.Linear(n_embedding, vocab_size)

    def forward(self, x):
        # Embeddings
        token_embeddings = self.token_embedding(x)
        positional_embeddings = self.positional_embedding(x)
        x = self.dropout(token_embeddings + positional_embeddings)

        # Encoder
        encoder_out = self.encoder(x)

        # Decoder
        decoder_out = self.decoder(x, encoder_out)

        # Output network
        res = self.dropout(self.fn(decoder_out))
        res = F.softmax(res)

        return res

class PositionalEncoding(nn.Module):
    """ Define the process to calculate the positional embeddings """
    def __init__(self, vocab_size, n_embeddings, max_len):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(0.5)
        pe = torch.zeros(max_len, n_embeddings)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

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

    def forward(self, x):
        x = x + self.dropout(self.pe[:x.size(0), :])
        return x

class EncoderBlock(nn.Module):
    def __init__(self, n_embedding, n_heads, n_layers, max_len, dropout=0.5):
        super(EncoderBlock, self).__init__()

        # Define Multi-headed attention
        self.multiheadattention = MultiHeadedAttention(n_embedding=n_embedding, n_heads=n_heads, n_layers=n_layers, max_len=max_len, dropout=dropout)
        # Define Feed Forward Network
        self.fcn = nn.Sequential(
            nn.Linear(n_embedding, 4*n_embedding),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(4*n_embedding, n_embedding),
            nn.Dropout(dropout),
        )
        # Define normalization
        self.norm = nn.LayerNorm(n_embedding)

    def forward(self, x):
        x = x + self.norm(self.multiheadattention(x))
        x = x + self.norm(self.fcn(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, n_embedding, n_heads, n_layers, max_len, dropout=0.5):
        super(DecoderBlock, self).__init__()

        # Define Multi-Headed attention
        self.multiheadattention = MultiHeadedAttention(n_embedding=n_embedding, n_heads=n_heads, n_layers=n_layers, max_len=max_len, dropout=dropout)
        # Define Feed Forward Network
        self.fcn = nn.Sequential(
            nn.Linear(n_embedding, 4*n_embedding),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(4*n_embedding, n_embedding),
            nn.Dropout(dropout),
        )
        # Define normalization
        self.norm = nn.LayerNorm(n_embedding)

    def forward(self, x, encoder_out):
        x = x + self.norm(self.multiheadattention(x, mask=True))
        x = x + self.norm(self.multiheadattention(x @ encoder_out, mask=True))
        x = x + self.norm(self.fcn(x))
        return x

class MultiHeadedAttention(nn.Module):
    """ Define the Multi-headed Attention Block with masking """
    def __init__(self, n_embedding, n_heads, n_layers, max_len, dropout=0.5):
        super(MultiHeadedAttention, self).__init__()

        self.n_heads = n_heads
        self.n_embeddings = n_embedding

        # Matrices for query, key and value vectors
        self.query = nn.Linear(n_embedding, n_embedding)
        self.key = nn.Linear(n_embedding, n_embedding)
        self.value = nn.Linear(n_embedding, n_embedding)
        # Projection for vector z
        self.projection = nn.Linear(n_embedding, n_embedding)

        # Normalization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_embedding)

        # Masking
        self.register_buffer(
            'mask', 
            torch.tril(torch.ones(size=(max_len, n_embedding))).view(max_len, 1, 1, n_embedding)
        )

    def forward(self, x, mask=None):
        B, T, C = x.size()
        # Get key, query and value vectors
        k = self.key(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        # Calculate and normalize score
        score = (q @ k.transpose(-1, -2)) / math.sqrt(self.n_embeddings)
        # Mask if requested
        if mask:
            score = score.masked_fill(self.mask == 0, float('-inf'))
        # Use softmax
        score = F.softmax(score, dim=-1)
        # Get final z vector
        z = (score @ v).transpose(1, 2).contiguous().view(B, T, C) 
        # Project z vector (best way to combine it) and apply dropout
        z = self.dropout(self.projection(z))

        return z
