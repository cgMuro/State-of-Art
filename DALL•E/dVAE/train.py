import torch
import torch.nn as nn
from dvae import dVAE
    
# ----------------------------------- DATA ----------------------------------- #
images = torch.randn(4, 3, 256, 256)


# ----------------------------------- PARAMETERS ----------------------------------- #

# Define loss
KL_LOSS_WEIGHT = 0
mse_loss = nn.MSELoss()

def get_loss(real, predicted, vocab_size):
    # logits = rearrange(logits, 'b n h w -> b (h w) n')
    log_qy = torch.nn.functional.log_softmax(real, dim=-1)
    log_uniform = torch.log(torch.tensor([1. / vocab_size], device=device))
    kl_div = torch.nn.functional.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target=True)
    return mse_loss(real, predicted) * (kl_div(real, predicted) *  KL_LOSS_WEIGHT)

# Define optimizer
optimizer = torch.optim.AdamW(dVAE.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=10e-8, weight_decay=0.999)

# Define cosine scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

# Define relaxation temperature
TEMPERATURE = 1.0
MIN_TEMPERATURE = 0.0625
TEMPERATURE_ANNELLING = 6.25e-6

# CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------------- TRAINING ----------------------------------- #

# Init model
VOCAB_SIZE = 8192
dVAE = dVAE(in_planes=1, hidden_planes=1, out_planes=1, blocks_per_group=1, temperature=TEMPERATURE, vocab_size=VOCAB_SIZE)

EPOCHS = 10
BATCH_SIZE = 8

# Model in training mode
dVAE.train().to(device)
# Zero model's gradient
dVAE.zero_grad()

print('Training dVAE...', '\n')
for epoch in range(EPOCHS):
    print('Epoch', epoch)
    total_loss = 0

    for image in images:
        image = image.to(device)

        res = dVAE(image)
        loss = get_loss(image, res, vocab_size=VOCAB_SIZE)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss

    print('Loss:', total_loss/len(images), '\n')

print('Training done.')
