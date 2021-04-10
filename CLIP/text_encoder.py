import torch
import torch.nn as nn
from transformer import Transformer
from utils import ModifiedLayerNorm

class TransformerTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,           # Define size of vocabulary
        max_length: int,           # Define sequence length
        width: int,                # Dimension for the embeddings 
        n_blocks: int,             # Number of blocks that compose the transformer (i.e. the depth)
        output_dim: int,           # Dimension of the output
        n_heads: int = 8,          # Number of heads for each multi-head attention layer
        head_dim: int = 64,        # Dimension of each multi-head layer
        dropout: float = 0.5       # Define dropout
    ):
        super().__init__()

        self.max_length = max_length
        self.n_blocks = n_blocks
        self.width = width

        # Define mask
        mask = self.get_mask()

        # Define embeddings
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=width)
        self.positional_embedding = nn.Parameter(torch.empty(max_length, width))

        # Define transformer
        self.transformer = Transformer(width=width, n_blocks=n_blocks, n_heads=n_heads, head_dim=head_dim, dropout=dropout, mask=mask)

        # Define modified layer normalization
        self.layernorm = ModifiedLayerNorm(width)

        # Define projection
        self.projection = nn.Parameter(torch.empty(width, output_dim))

        # Initialize parameters
        self.initialize_parameters()

    def get_mask(self):
        """ Function that creates the mask for the transformer """
        mask = torch.empty(self.max_length, self.max_length)  # Create empty mask
        mask.fill_(float('-inf'))  # Fill the entire mask with -inf
        mask.triu_(diagonal=1)  # Zero out the diagonal and the entries under the diagonal
        return mask

    def initialize_parameters(self):
        """ Function that handles the parameters initialization """
        # Init parameters for token and positional embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        # Init parameters for multihead attention and transformer's feedforward network
        proj_std = (self.width ** -0.5) * ((2 * self.n_blocks) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for attention, feedforward in self.transformer.layers:
            nn.init.normal_(attention.to_qkv.weight, std=attn_std)
            nn.init.normal_(attention.to_out.weight, std=proj_std)
            nn.init.normal_(feedforward.net[0].weight, std=fc_std)
            nn.init.normal_(feedforward.net[3].weight, std=proj_std)

        # Init parameters for projection
        nn.init.normal_(self.projection, std=(self.width ** -0.5))

    def forward(self, x: torch.Tensor):
        # Get index of the highest number along last dimension
        idx_max_n = x.argmax(dim=-1)
        
        # Apply token and positional embeddings
        x = self.token_embedding(x).type(x.dtype)   # shape = [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding(x).type(x.dtype)
        x = x.permute(1, 0, 2)

        # Apply transformer
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # shape = [batch_size, n_ctx, transformer.width]
        x = self.layernorm(x).type(x.dtype)

        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), idx_max_n] @ self.projection

        return x
