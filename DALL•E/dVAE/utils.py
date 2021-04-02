import torch

logit_laplance_epsilon: float = 0.1

# The decoder of the dVAE produces six feature maps representing the sufficient statistics of the logit-Laplace distribution
def map_input(x: torch.Tensor) -> torch.Tensor:
    # Before feeding an image into the dVAE encoder, we transform its values using φ : [0, 255] → (ε, 1 − ε)
    # The equation to do it is the following --> x → (1−2ε)x + ε
    return (1 - 2 * logit_laplance_epsilon) * x + logit_laplance_epsilon


# To reconstruct an image for manual inspection or computing metrics, we compute xˆ = φ−1 (sigmoid(μ)), where μ is given by the first three feature maps output by the dVAE decoder
def unmap_input(x: torch.Tensor) -> torch.Tensor:
    # The equation to do it is the following --> (x - ε) / (1 - 2*ε)
    return torch.clamp(((x - logit_laplance_epsilon) / (1 - 2 * logit_laplance_epsilon)), min=0, max=1)  # Clamps all elements in input into the range [ min, max ]
