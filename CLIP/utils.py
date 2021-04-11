import torch
import torch.nn as nn

class ModifiedLayerNorm(nn.LayerNorm):
    """ Subclass torch's LayerNorm to handle fp16 """
    def forward(self, x: torch.Tensor):
        orgin_type = x.dtype  # Store x's orginal type
        ret = super().forward(x.type(torch.float32))  # Apply nn.LayerNorm on x after changing its type to torch.float32
        return ret.type(orgin_type)  # Return x into its original type
