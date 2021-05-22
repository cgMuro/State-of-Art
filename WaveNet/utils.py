import numpy as np

# MU-LAW
# Encoding
def mu_law_encoding(data, mu=255):
    return np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
# Decoding/Expansion
def mu_law_decoding(data, mu=255):
    return np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
