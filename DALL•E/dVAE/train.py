import torch
import torch.nn as nn
import numpy as np
import einops
from dvae import dVAE
from utils import preprocess_image, map_image, unmap_image


# ----------------------------------- DATA ----------------------------------- #
images = [torch.randn(3, 256, 256)]


# ----------------------------------- TRAINING ----------------------------------- #

# TRAINING PARAMETERS
EPOCHS = 10
BATCH_SIZE = 8
VOCAB_SIZE = 8192
# Define relaxation temperature
TEMPERATURE = 1.0
MIN_TEMPERATURE = 0.0625
TEMPERATURE_ANNELLING = 6.25e-6
# Define KL loss weight
KL_LOSS_WEIGHT_INITIAL = 0
KL_LOSS_WEIGHT_MAX = 6.6
KL_LOSS_WEIGHT_UPDATES = 5000

IMAGE_SIZE = 256
IN_PLANES = 3
HIDDEN_PLANES = 256
OUT_PLANES = 3
BLOCKS_PER_GROUP = 1

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Init model
dvae = dVAE(in_planes=IN_PLANES, hidden_planes=HIDDEN_PLANES, out_planes=OUT_PLANES, blocks_per_group=BLOCKS_PER_GROUP, vocab_size=VOCAB_SIZE)
# Move model on CUDA, if available, else CPU
dvae = dvae.to(device=DEVICE)
# Model in training mode
dvae.train()

# Define optimizer
optimizer = torch.optim.AdamW(dvae.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=10e-8, weight_decay=0.999)

# Define learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

# Define loss
mse_loss = nn.MSELoss()

def get_loss(ground_truth, enc_out, dec_out, vocab_size, kl_loss_weight):
    # Rearrange encoder's output
    enc_out = einops.rearrange(enc_out, 'b n h w -> b (h w) n')
    # Apply log softmax
    log_qy = torch.nn.functional.log_softmax(enc_out, dim=-1)
    # Get log uniform
    log_uniform = torch.log(torch.Tensor([1. / vocab_size], device=DEVICE))
    # Apply KL divergence
    kl_div = torch.nn.functional.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target=True)
    # Return total loss
    return mse_loss(ground_truth, dec_out) * (kl_div * kl_loss_weight)



# START TRAINING
print('Training dVAE...', '\n')

# Init temperature and KL weight
temperature = TEMPERATURE
kl_loss_weight = KL_LOSS_WEIGHT_INITIAL

for epoch in range(EPOCHS):
    print('Epoch', epoch)
    # Init total epoch loss
    total_loss = 0

    for i, image in enumerate(images):
        # Zero out gradients
        optimizer.zero_grad()

        # Set device and preprocess image
        image = image.to(DEVICE)
        image = preprocess_image(image, target_img_size=IMAGE_SIZE)

        # Get prediction from model
        encoder_out, decoder_out = dvae(image, temperature)

        # Calculate loss
        loss = get_loss(image, encoder_out, decoder_out, vocab_size=VOCAB_SIZE, kl_loss_weight=kl_loss_weight)
        if i % 100: print('\n', 'Loss', loss, '\n')
        # Calculate gradients
        loss.backward()
        # Update parameters
        optimizer.step()
        # Update scheduler
        lr_scheduler.step()
        # Update temperature and KL weight
        temperature = np.maximum(temperature * np.exp(-TEMPERATURE_ANNELLING * i), MIN_TEMPERATURE)
        kl_loss_weight = KL_LOSS_WEIGHT_INITIAL + 0.5 * (KL_LOSS_WEIGHT_MAX - KL_LOSS_WEIGHT_INITIAL)*(1 + np.cos((i * np.pi / KL_LOSS_WEIGHT_UPDATES) * np.pi))
        # Update total epoch loss
        total_loss += loss

    print('TOTAL LOSS:', total_loss/len(images), '\n\n')

print('Training done.')
