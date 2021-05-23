import numpy as np

# MU-LAW

# Encoding
def mu_law_encoding(data, num_classes=255):
    return np.sign(data) * np.log(1 + num_classes * np.abs(data)) / np.log(num_classes + 1)
# Decoding/Expansion
def mu_law_decoding(data, num_classes=255):
    return np.sign(data) * (np.exp(np.abs(data) * np.log(num_classes + 1)) - 1) / num_classes
