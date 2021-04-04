import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

# Define entire discrete Variational Auto Encoder
class dVAE(nn.Module):
    def __init__(self, in_planes: int, hidden_planes: int, out_planes: int, blocks_per_group: int, temperature : float, vocab_size: int = 8192):
        self.temp = temperature

        self.encoder = Encoder(in_planes, hidden_planes, out_planes, blocks_per_group, vocab_size)
        self.decoder = Decoder(in_planes, hidden_planes, out_planes, blocks_per_group, vocab_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        out = self.encoder(x)
        # Processing
        soft_one_hot = nn.functional.gumbel_softmax(out, tau=temp, hard=False, dim=1)
        sampled = torch.einsum('b n h w, n d -> b d h w', soft_one_hot) # self.codebook.weight(?)
        # Decoder
        out = self.decoder(sampled)

        return out
