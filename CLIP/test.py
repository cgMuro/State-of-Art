import os
import torch
import numpy as np
from PIL import Image
from clip import CLIP
from utils import augment_image
from tokenizer import SimpleTokenizer


MODEL_PATH = ''                                            # File path to saved model
DATA_DIR = ''                                              # File path to data directory
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'    # Set device (either cuda or cpu)
tokenizer = SimpleTokenizer()                              # Init tokenizer

# ----------------------------------- DATA ----------------------------------- #

def process_data(data_dir: str, descriptions: dict, image_size: int, device: torch.DeviceObjType, token_max_length: int = 76):
    ''' Process data given the directory containing the images and a "descriptions" dictionary that maps each image filename to a caption/description '''
    # Init data arrays
    images = []
    texts = []
    # Init image mean and standard deviation
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)

    # Iterate over files in passed directory
    for filename in [filename for filename in os.listdir(data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
        # Get filename
        name = os.path.splitext(filename)[0]

        # Check if the name is in the descriptions passed
        if name not in descriptions: continue

        # Get and process image
        image = augment_image(Image.open(os.path.join(data_dir, filename)).convert("RGB"), image_size)
        images.append(image)
        # Get description
        texts.append(descriptions[name])

    # Normalize images
    image_input = torch.tensor(np.stack(images)).to(device)
    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]

    # Tokenize texts
    text_tokens = [tokenizer.encode(f'This is {desc}') for desc in texts]

    text_input = torch.zeros(len(text_tokens), token_max_length, dtype=torch.long)
    sot_token = tokenizer.encoder['<|startoftext|>']
    eot_token = tokenizer.encoder['<|endoftext|>']

    for i, tokens in enumerate(text_tokens):
        tokens = [sot_token] + tokens + [eot_token]
        text_input[i, :len(tokens)] = torch.tensor(tokens)
    
    text_input = text_input.to(DEVICE)

    return image_input, text_input


# ----------------------------------- VALIDATION ----------------------------------- #

# Get model
model = CLIP()
# Load trained model
model = model.load_pretrained_from_file(MODEL_PATH)
# Move model to device
model = model.to(DEVICE)
# Model in evaluation mode
model.eval()


def get_cosine_similarities(model: torch.nn.Module, data_dir: str, descriptions: dict, device: torch.DeviceObjType):
    ''' Given a pretrained model, data directory and descriptions of the images in the directory, returns cosine similarities '''
    # Get data
    images, texts = process_data(data_dir=data_dir, descriptions=descriptions, image_size=model.image_size, device=device, token_max_length=model.max_length)

    # Get image and text features from model
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

    # Calculate cosine similarity
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    return similarity


def predict(model: torch.nn.Module, images: torch.Tensor, texts: torch.Tensor, device: torch.DeviceObjType, top_k_return: int = 5,):
    ''' Takes in a pretrained model, the processed images and texts, model's device, and returns the number ("top_k_return") of top probabilities and labels '''
    # Move tensors to device
    images = images.to(device)
    texts = texts.to(device)

    # Get prediction from model
    with torch.no_grad():
        # Get image features
        image_features = model.encode_image(images)
        # Get text features and normalize
        text_features = model.encode_text(texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Get text probabilities
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    # Get top 5 probabilities and labels
    top_probs, top_labels = text_probs.cpu().topk(top_k_return, dim=-1)

    return top_probs, top_labels



# Dummy data for predict function
images = []
texts = []
for i in range(5):
    images.append(augment_image(torch.rand(3, 400, 400), 224))
    texts.append(torch.randint(low=0, high=76, size=(76,), dtype=torch.long))

images = torch.stack(images)
texts = torch.stack(texts)

top_probs, top_labels = predict(model=model, images=images, texts=texts, device=DEVICE, top_k_return=5)

print(top_probs)
print(top_labels)
