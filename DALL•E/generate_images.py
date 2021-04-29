import torch
from dall_e import DALLE
from dVAE.utils import preprocess_image


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


BATCH_SIZE: int = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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

# Dummy data
img = torch.stack([
        preprocess_image(image, target_img_size=IMAGE_SIZE) for image in torch.rand(BATCH_SIZE, 3, 400, 400, device=DEVICE)
], dim=1)[0]
text = torch.randint(low=0, high=TRANSFORMER_VOCAB_SIZE, size=(BATCH_SIZE, MAX_LENGTH), device=DEVICE)

# Generate images with model
res = model.generate_images(text, img)

print(res)
