import torch
import einops
from dall_e import DALLE
from dVAE.utils import preprocess_image
from tokenizer import SimpleTokenizer
from utils import tokenize

# TRAINING PARAMETERS
EPOCHS: int = 1
BATCH_SIZE: int = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WEIGHT_DECAY: float = 4.5 * 10e-2
ADAM_BETA_1: float = 0.9
ADAM_BETA_2: float = 0.96
ADAM_EPS: float = 10e-8
LEARNING_RATE: float = 5 * 10e-5
CLIP_THRESHOLD: int = 4

# dVAE parameters
IN_PLANES: int = 3
HIDDEN_PLANES: int = 256
OUT_PLANES: int = 3
BLOCKS_PER_GROUP: int = 1
DVAE_VOCAB_SIZE: int = 8192
IMAGE_SIZE: int = 256

# transformer parameters
N_BLOCK: int = 1
N_HEADS: int = 1
HEAD_DIM: int = 1
MAX_LENGTH: int = 256
TRANSFORMER_VOCAB_SIZE: int = 16384
DROPOUT: float = 0.5


# Data
data = [
    [
        torch.rand(BATCH_SIZE, 3, 400, 400),
        torch.randint(low=0, high=TRANSFORMER_VOCAB_SIZE, size=(BATCH_SIZE, MAX_LENGTH))
    ]
]

# Get tokenizer
tokenizer = SimpleTokenizer()

# Get model
model = DALLE(
    in_planes=IN_PLANES, 
    hidden_planes=HIDDEN_PLANES, 
    out_planes=OUT_PLANES, 
    blocks_per_group=BLOCKS_PER_GROUP, 
    dVAE_vocab_size=DVAE_VOCAB_SIZE, 
    n_block=N_BLOCK, 
    n_heads=N_HEADS, 
    head_dim=HEAD_DIM, 
    max_length=MAX_LENGTH, 
    transformer_vocab_size=TRANSFORMER_VOCAB_SIZE, 
    dropout=DROPOUT
)
# Move model on CUDA, if available, else CPU
model = model.to(device=DEVICE)
# Model in training mode
model.train()

# Loss
criterion_image = torch.nn.CrossEntropyLoss()
criterion_text = torch.nn.CrossEntropyLoss()
# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)
# Scheduler
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

# START TRAINING
print('\n', 'Training DALL•E...', '\n')

for epoch in range(EPOCHS):
    print('Epoch', epoch)
    # Init total epoch loss
    total_loss = 0

    for images, texts in data:
        # Zero out gradients
        optimizer.zero_grad()

        # Set device and preprocess image
        images = images.to(DEVICE)
        images = torch.stack([preprocess_image(image, target_img_size=IMAGE_SIZE) for image in images], dim=1)[0]

        # Tokenize text and set device
        texts = tokenize(tokenizer, texts, max_length=MAX_LENGTH)
        texts = texts.to(DEVICE)

        # Get prediction from model
        images_embedding, logits = model(image=images, text=texts)
        
        # Calculate loss
        labels = torch.cat((texts[:, :], images_embedding + MAX_LENGTH), dim=1)  # Concatenate text and image ground truths 
        # Image loss
        loss_image = criterion_image(logits[:, :, MAX_LENGTH:], labels[:, MAX_LENGTH:])
        # Text loss
        loss_text = criterion_text(logits[:, :, :MAX_LENGTH], labels[:, :MAX_LENGTH])
        # Total loss
        total_loss = (0.125 * loss_text) + (0.875 * loss_image)
        print('LOSS:', total_loss)

        # Calculate gradients
        total_loss.backward()
        # Apply gradient clipping and update parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_THRESHOLD)
        optimizer.step()
        # Update scheduler
        lr_scheduler.step()

    print('TOTAL LOSS:', total_loss/len(images), '\n\n')

print('Training done.')
