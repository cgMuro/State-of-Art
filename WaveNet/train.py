import torch
from model import WaveNet
from data import WaveNet_Dataset, AudioLoader


# ----------------------------------- DATA ----------------------------------- #

# Data parameters
FILE_LIST: str = ''
X_LEN: int = 1
Y_LEN: int = 1
NUM_CLASSES: int = 256
STORE_TRACKS: bool = True
BATCH_SIZE: int = 32
NUM_WORKERS: int = 1
BITRATE: int = 16
TWOS_COMP: bool = True
ENCODER = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Create dataset
dataset = WaveNet_Dataset(
    track_list=FILE_LIST, 
    x_len=X_LEN, 
    y_len=Y_LEN, 
    bitrate=BITRATE, 
    twos_comp=TWOS_COMP, 
    num_classes=NUM_CLASSES,
    store_tracks=STORE_TRACKS,
    encoder=ENCODER
)
# Create dataloader
dataloader = AudioLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE, 
    num_workers=NUM_WORKERS
)

# ----------------------------------- TRAIN ----------------------------------- #

# MODEL
HIDDEN: int = 16
KERNEL_SIZE: int = 1
N_BLOCKS: int = 1

model = WaveNet(num_classes=NUM_CLASSES, hidden=HIDDEN, kernel_size=KERNEL_SIZE, n_blocks=N_BLOCKS)
# Move model on CUDA, if available, else CPU
model = model.to(device=DEVICE)
# Model in training mode
model.train()


# Training parameters
EPOCHS = 10
LEARNING_RATE: float = 0.001
WEIGHT_DECAY: float = 0.0

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Start training
print('\n', 'Training WaveNet...', '\n')

for epoch in range(EPOCHS):
    print('Epoch', epoch)

    # Init total epoch loss
    total_loss = 0

    for input, target in dataloader:
        # Zero out gradients
        optimizer.zero_grad()

        # Move data to device
        input = input.to(device=DEVICE)
        target = target.to(device=DEVICE)

        # Get model prediction
        res = model(input)

        # Loss
        loss = torch.nn.functional.cross_entropy(res, target)
        total_loss += loss

        # Calculate gradients
        loss.backward()

        # Update parameters
        optimizer.step()

    print('LOSS:', total_loss/len(dataloader), '\n\n')

print('Training done.')
