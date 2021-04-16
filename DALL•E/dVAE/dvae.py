import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

# Define entire discrete Variational Auto Encoder
class dVAE(nn.Module):
    def __init__(
        self, 
        in_planes: int,          # Input channels
        hidden_planes: int,      # Hidden units
        out_planes: int,         # Output channels
        blocks_per_group: int,   # Number of ResNet's bottleneck blocks for each group
        vocab_size: int = 8192   # Size of vocaboulary
    ):
        super().__init__()

        # Encoder
        self.encoder = Encoder(in_planes=in_planes, hidden_planes=hidden_planes, out_planes=out_planes, blocks_per_group=blocks_per_group, vocab_size=vocab_size)
        # Decoder
        self.decoder = Decoder(in_planes=in_planes, hidden_planes=hidden_planes, out_planes=out_planes, blocks_per_group=blocks_per_group, vocab_size=vocab_size)

    def forward(self, x: torch.Tensor, temperature: float) -> torch.Tensor:
        # Pass input through encoder
        encoder_out = self.encoder(x)
        # To optimize the distribution over the 32 × 32 image tokens generated by the dVAE encoder we use the gumbel-softmax relaxation, where the relaxation becomes tight as the temperature τ → 0
        out = nn.functional.gumbel_softmax(encoder_out, tau=temperature, hard=False, dim=1)
        # Pass input through decoder
        decoder_out = self.decoder(out)

        return encoder_out, decoder_out
