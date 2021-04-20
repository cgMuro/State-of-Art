import torch
import torch.nn as nn
from transformer import Transformer
from utils import ModifiedLayerNorm

class TransformerTextEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int,           # Dimension of the output [possible values from the paper: 512, 768]
        vocab_size: int = 49152,   # Define size of vocabulary
        max_length: int = 76,      # Define maximum sequence length
        width: int = 512,          # Dimension for the embeddings [possible values from the paper: 512, 768]
        n_blocks: int = 12,        # Number of blocks that compose the transformer (i.e. the depth) [possible values from the paper: 12, 16]
        n_heads: int = 8,          # Number of heads for each multi-head attention layer [possible values from the paper: 8, 12]
        head_dim: int = 64,        # Dimension of each multi-head layer
        dropout: float = 0.5,      # Define dropout
        tensor_type: torch.TensorType = torch.float # Define type of tensor to convert input data to
    ):
        super().__init__()

        self.max_length = max_length
        self.n_blocks = n_blocks
        self.width = width
        self.tensor_type = tensor_type

        # Define token and positional embeddings
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=width)
        self.positional_embedding = nn.Parameter(torch.empty(max_length, width))

        # Define transformer
        self.transformer = Transformer(n_embeddings=width, n_blocks=n_blocks, n_heads=n_heads, head_dim=head_dim, dropout=dropout)

        # Define modified layer normalization
        self.layernorm = ModifiedLayerNorm(width)

        # Define projection
        self.projection = nn.Parameter(torch.empty(width, output_dim))

        # Initialize parameters
        self.initialize_parameters()

    def get_mask(self, batch: int):
        """ Function that creates the mask for the transformer """
        mask = torch.empty(self.max_length, batch)  # Create empty mask
        # mask.fill_(float('-inf'))  # Fill the entire mask with -inf
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
            nn.init.normal_(attention.to_out[0].weight, std=proj_std)
            nn.init.normal_(feedforward.net[0].weight, std=fc_std)
            nn.init.normal_(feedforward.net[3].weight, std=proj_std)

        # Init parameters for projection
        nn.init.normal_(self.projection, std=(self.width ** -0.5))

    def forward(self, x: torch.Tensor):
        # Get index of the highest number along last dimension
        idx_max_n = x.argmax(dim=-1)
        
        # Apply token and positional embeddings
        x = self.token_embedding(x).type(self.tensor_type)   # shape = [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.tensor_type)
        x = x.permute(1, 0, 2)

        # Define mask
        mask = self.get_mask(batch=x.size()[1])
        
        # Apply transformer, permute and normalize
        x = self.transformer(x, mask=mask)
        x = x.permute(1, 0, 2)  # shape = [batch_size, n_ctx, transformer.width]
        x = self.layernorm(x).type(self.tensor_type)

        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), idx_max_n] @ self.projection

        return x
