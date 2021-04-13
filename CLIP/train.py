import torch
from clip import CLIP
from tokenizer import SimpleTokenizer
from utils import augment_image, tokenize


# DATA

data = [[torch.rand(3, 400, 400), torch.randint(low=0, high=76, size=(1, 76), dtype=torch.long)]]

# Get tokenizer
tokenizer = SimpleTokenizer()


# TRAINING PARAMETERS
BATCH_SIZE: int = 32768
EPOCHS: int = 32
INITIAL_TEMPERATURE: float = 0.07
MAX_TEMPERATURE: float = 100.0
WEIGHT_DECAY: float = 0.2
WARMUP_ITERATIONS: int = 2000
ADAM_BETA_1: float = 0.9
ADAM_BETA_2: float = 0.98
ADAM_EPS: float = 10e-6
LEARNING_RATE: float = 5 * 10e-4

EMBEDDING_DIMENSION: int = 512
IMAGE_SIZE: int = 224
PATCH_SIZE: int = 32
VISION_BLOCKS: int = 12   
VISION_WIDTH: int = 768           # Dimension for the image embeddings [possible values from the paper: 768, 1024]
VISION_HEADS: int = 12            # Number of heads for each multihead attention layer in the vision transformer [possible values from the paper: 12, 16]
MAX_LENGTH: int = 76              # Maximum sequence length
VOCABULARY_SIZE: int = 49408
TEXT_BLOCKS: int = 12             # Number of blocks that compose the transformer (i.e. the depth) [possible values from the paper: 12, 16]
TEXT_WIDTH: int = 512             # Dimension for the text embeddings [possible values from the paper: 512, 768]
TEXT_HEADS: int =  8

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get model
model = CLIP(
    emb_dim=EMBEDDING_DIMENSION,
    temperature=(INITIAL_TEMPERATURE, MAX_TEMPERATURE),
    image_size=IMAGE_SIZE, 
    patch_size=PATCH_SIZE, 
    vision_blocks=VISION_BLOCKS, 
    vision_width=VISION_WIDTH, 
    vision_heads=VISION_HEADS, 
    max_length=MAX_LENGTH, 
    vocab_size=VOCABULARY_SIZE, 
    text_blocks=TEXT_BLOCKS,
    text_width=TEXT_WIDTH,
    text_heads=TEXT_HEADS
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
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, WARMUP_ITERATIONS)


for epoch in range(EPOCHS):
    for images, texts in data:
        # Zero out gradients
        optimizer.zero_grad()

        # Augment image and set device
        images = images.to(DEVICE)
        images = augment_image(images, size=IMAGE_SIZE)

        # Tokenize text and set device
        texts = tokenize(tokenizer, texts, max_length=MAX_LENGTH)
        texts = texts.to(DEVICE)

        # Get prediction by model
        logits_per_image, logits_per_text = model(images, texts)
        
        # Calculate loss
        labels = torch.arange(len(logits_per_image)).to(DEVICE)
        loss_image = criterion_image(logits_per_image, labels)
        loss_text = criterion_text(logits_per_text, labels)
        total_loss = (loss_image + loss_text) / 2
        print(total_loss)

        # Calculate gradients
        total_loss.backward()
        # Update parameters
        optimizer.step()
        # Update scheduler
        scheduler.step()
