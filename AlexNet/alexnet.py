# https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
# https://pytorch.org/hub/pytorch_vision_alexnet/

import urllib
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# AlexNet model
class AlexNet(nn.Module):
    ''' AlexNet architecture implemented in PyTorch '''
    def __init__(self, num_classes : int = 1000) -> None:
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # First Convolutional Layer + Maxpooling
            nn.Conv2d(  # Type of layer that is used to extract features from an image
                in_channels=3,   # Number of channels in the input image. For example in this case it's 3 because the image has colors (RGB -> 3 channels)
                out_channels=64,    # Number of channels produced by the convolutional layer. Can be interpreted as the number of filters used to map the features of the image
                kernel_size=(11, 11),  # Size of the matrix that defines the way pixels are going to be summed together
                stride=4,  # Size of the movement the kernel matrix will do over the image
                padding=2  # Padding of 2 added to the input to make it even
            ),
            nn.ReLU(inplace=True), # Applies the rectified linear unit function (ReLU) in place
            nn.MaxPool2d(  # Type of layer that is used to reduce the size of image tensor
                kernel_size=(3, 3),  # Size of the matrix that defines the region from which pixels are going to be picked
                stride=2  # Size of the movement the kernel matrix will do over the image
            ),
            # Second Convolutional Layer + Maxpooling
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # Third Convolutional Layer
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Fourth Convolutional Layer
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Fifth Convolutional Layer + Maxpooling
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
        )

        # Average Pooling downsamples the data by taking the average over a patch of values, it's adaptive because you define the output size and it will automatically choose kernel and stride values
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(
                in_features=256 * 6 * 6,  # Number of input features. In this case:  256 -> output of last Conv2d, 6*6 -> output size of Adaptive Average Pooling
                out_features=4096  # Number of output features
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes)
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(input=x, start_dim=1) # Flattens the first dimension of the input tensor and therefore returns a 1D Tensor
        x = self.classifier(x)

        return x



# Load pretrained AlexNet model from PyTorch
model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
# Put model in evaluation mode
model.eval()


# Dowload an example from PyTorch website
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try:
    urllib.request.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)

# File into PIL image
input_image = Image.open(filename)
# Compose several transformations needed for preprocessing
preprocess = transforms.Compose([
    transforms.Resize(size=256),    #  Resizes the given image to the size passed
    transforms.CenterCrop(size=224), # Crops the given image at the center with a desired output size
    transforms.ToTensor(),  # Converts PIL image or numpy.ndarray to tensor
    transforms.Normalize(   # Normalizes a tensor image with mean and standard deviation
        mean=[0.485, 0.456, 0.406],  # sequence of mean for each channel
        std=[0.229, 0.224, 0.225]  # sequence of standard deviations for each channel
    )
])
# Apply the preprocessing to the image
input_tensor = preprocess(input_image)
# Create a mini-batch as expected by the model
input_batch = input_tensor.unsqueeze(0)

# Move input and model in GPU if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

# Prediction
with torch.no_grad():  # Disables the calculation of gradients
    output = model(input_batch)  # Get prediction from model

print(output[0])

# To get probabilities run softmax function on output
probabilities = nn.functional.softmax(output[0], dim=0)
print(probabilities)
