import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from clip import CLIP
from utils import augment_image
from tokenizer import SimpleTokenizer


MODEL_PATH: str = ''                                                           # File path to saved model
DATA_DIR: str = ''                                                             # File path to data directory
DEVICE: torch.DeviceObjType = 'cuda' if torch.cuda.is_available() else 'cpu'   # Set device (either cuda or cpu)
tokenizer: SimpleTokenizer = SimpleTokenizer()                                 # Init tokenizer


# ----------------------------------- DATA ----------------------------------- #

def process_data(data_dir: str, descriptions: dict, image_size: int,  tokenizer: SimpleTokenizer, device: torch.DeviceObjType, token_max_length: int = 76):
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


def get_cosine_similarities(model: torch.nn.Module, data_dir: str, descriptions: dict, tokenizer: SimpleTokenizer, device: torch.DeviceObjType):
    ''' Given a pretrained model, data directory and descriptions of the images in the directory, returns cosine similarities '''
    # Get data
    images, texts = process_data(data_dir=data_dir, descriptions=descriptions, image_size=model.image_size, tokenizer=tokenizer, device=device, token_max_length=model.max_length)

    # Get image and text features from model
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

    # Calculate cosine similarity
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    return similarity


def test_from_directory(model_path: str, data_dir: str, descriptions: dict, tokenizer: SimpleTokenizer, device: torch.DeviceObjType, top_k_returns: int = 5, model_parameters: dict = None):
    ''' Test a trained model using data contained in a directory '''
    # MODEL
    if model_parameters is not None:
        model = CLIP(**model_parameters)
    else:
        model = CLIP()
    # Load trained model
    model = model.load_pretrained_from_file(MODEL_PATH)
    # Move model to device
    model = model.to(device)
    # Model in evaluation mode
    model.eval()
    
    # DATA
    images, texts = process_data(data_dir=data_dir, descriptions=descriptions, image_size=model.image_size, tokenizer=tokenizer, device=device, token_max_length=model.max_length)
    data_loader = torch.utils.data.DataLoader(torch.stack(images, texts), batch_size=32, num_workers=16)
    # Build zero shot weights
    with torch.no_grad():
        zeroshot_weights = []
        for desc in tqdm(descriptions.values()):
            texts = tokenizer.encode(desc).to(device)    # Tokenizer
            class_embeddings = model.encode_text(texts) # Embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    # PREDICT
    with torch.no_grad():
        top1 = 0.0
        top5 = 0.0
        counter = 0.0
        for idx, (images, target) in enumerate(tqdm(data_loader)):
            # Get image features
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights
            # Get accuracy
            predictions = logits.topk(top_k_returns, 1, True, True)[1].t()
            correct = predictions.eq(target.view(1, -1).expand_as(predictions))
            acc1, acc5 = [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in top_k_returns]
            # Update stats
            top1 += acc1
            top5 += acc5
            counter += images.size(0)
    
    top1 = (top1 / counter) * 100
    top5 = (top5 / counter) * 100 

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")


def test_from_dataset(model_path: str, data, classnames: list, templates: list, tokenizer: SimpleTokenizer, device: torch.DeviceObjType, top_k_returns: int = 5, model_parameters: dict = None):
    ''' Test a trained model using data contained in a dataset '''
    # MODEL
    if model_parameters is not None:
        model = CLIP(**model_parameters)
    else:
        model = CLIP()
    # Load trained model
    model = model.load_pretrained_from_file(MODEL_PATH)
    # Move model to device
    model = model.to(device)
    # Model in evaluation mode
    model.eval()
    
    # DATA
    data_loader = torch.utils.data.DataLoader(data, batch_size=32, num_workers=16)

    # Init image mean and standard deviation
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)

    # Build zero shot weights
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # Format template with class
            texts = tokenizer.encode(texts).to(device)                      # Tokenize
            class_embeddings = model.encode_text(texts)                     # Embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    # PREDICT
    with torch.no_grad():
        top1 = 0.0
        top5 = 0.0
        counter = 0.0
        for idx, (images, target) in enumerate(tqdm(data_loader)):
            # Normalize images
            images = [augment_image(image, model.image_size) for image in images]
            images -= image_mean[:, None, None]
            images /= image_std[:, None, None]
            # Get image features
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights
            # Get accuracy
            predictions = logits.topk(top_k_returns, 1, True, True)[1].t()
            correct = predictions.eq(target.view(1, -1).expand_as(predictions))
            acc1, acc5 = [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in top_k_returns]
            # Update stats
            top1 += acc1
            top5 += acc5
            counter += images.size(0)
    
    top1 = (top1 / counter) * 100
    top5 = (top5 / counter) * 100 

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")


def predict(model: torch.nn.Module, images: torch.Tensor, texts: torch.Tensor, device: torch.DeviceObjType, top_k_returns: int = 5,):
    ''' Takes in a pretrained model, the processed images and texts, model's device, and returns the number ("top_k_returns") of top probabilities and labels '''
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
    top_probs, top_labels = text_probs.cpu().topk(top_k_returns, dim=-1)

    return top_probs, top_labels


# Dummy data for predict function
images = []
texts = []
for i in range(5):
    images.append(augment_image(torch.rand(3, 400, 400), 224))
    texts.append(torch.randint(low=0, high=76, size=(76,), dtype=torch.long))

images = torch.stack(images)
texts = torch.stack(texts)

top_probs, top_labels = predict(model=model, images=images, texts=texts, device=DEVICE, top_k_returns=5)

print(top_probs)
print(top_labels)
