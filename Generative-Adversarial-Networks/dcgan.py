# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

# ----------------------------------- GAN ----------------------------------- #
# GAN (Generative Adversarial Networks) is an architecture used to teach a model to generate new data from the same distribution of the given training data.
# GANs are made of 2 models: 
    # a generator -> which produces fake data that look similar to the training data
    # a discriminator -> which tries to identify if the given example is fake or not
# Over time, by working one against the other, the discriminator will become better at recognizing fakes and the generator will improves its creations.
# Ideally, the equilibrium is reached when the generator produces perfect fakes and the discriminator is left to always guess with 50% of confidence.

# ----------------------------------- DCGAN ----------------------------------- #
# A DCGAN (Deep Convolutional Generative Adversarial Networks) is a direct extension of the GAN architecture, in which:
    # the generator is comprised of convolutional-transpose layers, batch norm layers, and ReLU activations
    # the discriminator is made up of strided convolution layers, batch norm layers, and LeakyReLU activations



from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manual_seed = 999
print('Random Seed:', manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)


# Root directory for dataset
dataroot = 'data/celeba'
# Number of workers for dataloader
workers = 2
# Batch size during training
batch_size = 128
# Dimension for resizing images
image_size = 64
# Number of channels in the training images
nc = 3
# Size of z latent vector (i.e. size of generator input) 
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 5
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available (use 0 for CPU model)
ngpu = 1


# ----------------------------------- DATA ----------------------------------- #
# Dataset -> here http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html or here https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg

# Create dataset
dataset = dset.ImageFolder(
    root=dataroot,
    transform=transforms.Compose([
        transforms.Resize(image_size),  # Resize image to the passed parameter
        transforms.CenterCrop(image_size),  # Crops image to the center and keeping the size equal to the passed parameter
        transforms.ToTensor(),     # Transform image to tensor
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize the tensor image using the passed mean and strandard deviation
    ])
)

# Create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset=dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=workers
)

# Decide which device we want to run on
device = torch.device('cuda:0', if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis('off')
plt.title('Training Images')
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1,2,0)))


# ----------------------------------- MODEL ----------------------------------- #

# WEIGHT INITIALIZATION #
# All model weights shall be randomly initialized from a normal distribution with mean=0 and stdev=0.02
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weigh.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# GENERATOR #

# The generator maps the latent space vector (z) to data-space. 
# Since our data are images, converting z to data-space means creating a RGB image with the same size as the training images (i.e. 3x64x64). 
# This is accomplished through a series of strided two dimensional convolutional transpose layers, each paired with 
# a 2d batch normalization layer (which is critical and helps with the flow of gradients during training) and a relu activation. 
# Then the output of the generator is fed through a tanh function to squeeze it to the range [−1,1].

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is Z going into a convolution
            nn.ConvTranspose2d(     # The Convolutional Transposed layer is used for upsampling -> it generates an output feature map that has a spatial dimension greater than that of the input feature map
                in_channels=nz,     # Number of channels in the input image, 100 in this case
                out_channels=ngf*8, # Number of output channels produced by the convolution
                kernel_size=4,      # Size of the kernel
                stride=1,           # Stride of the convolution
                padding=0,          # Padding added to both sides of each dimension in the input
                bias=False          # Add a learnable bias to the output
            ),
            nn.BatchNorm2d(num_features=ngf*8),
            nn.ReLU(inplace=True)
            
            # State size (ngf*8) * 4 * 4
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),

            # State size (ngf*8) * 8 * 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),

            # State size (ngf*2) * 16 * 16
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # State size (ngf) * 32 * 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # State size (nc) * 64 * 64
        )

    def forward(self, input):
        return self.main(input)

# Instantiate the generator and apply the weights_init function
netG = Generator(ngpu=ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Randomly initialize all weights to mean=0 and stdev=0.2
netG.apply(weights_init)

print(netG)


# DISCRIMINATOR #
# The discriminator is a binary classification network that takes an image as input and outputs a scalar probability that the input image is real (as opposed to fake). 
# Here, the discriminator takes a 3x64x64 input image, processes it through a series of Conv2d, BatchNorm2d, and LeakyReLU layers, and outputs the final probability through a Sigmoid activation function. 
# This architecture can be extended with more layers if necessary for the problem.
# It's good practice to use strided convolution rather than pooling to downsample because it lets the network learn its own pooling function. 
# Also batch norm and leaky relu functions promote healthy gradient flow which is critical for the learning process of both generator and discriminator models.

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is (nc) * 64 * 64
            nn.Conv2d(
                in_channels=nc, 
                out_channels=ndf, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # State size (ndf) * 32 * 32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            # State size (ndf*2) * 16 * 16
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            # State size (ndf*4) * 8 * 8
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            # State size (ndf*8) * 4 * 4
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.main(input)

# Instantiate the discriminator and apply the weights_init function
netD = Discriminator(ngpu=ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Randomly initialize all weights to mean=0 and stdev=0.2
netD.apply(weights_init)

print(netD)


# LOSS FUNCTIONS AND OPTIMIZERS #

# Init Binary Cross Entropy Loss function
criterion = nn.BCELoss()

# Create batch of latent vectors -> will be used to visualize the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Init real and fake labels
real_label = 1
fake_label = 0

# Init optimizers
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))



# ----------------------------------- TRAINING ----------------------------------- #

# 1. Discriminator training
#       The goal is to maximize probability of correctly classifying a given input as real or fake -> max[ log(D(x)) + log(1-D(G(z))) ]
#       To do it: 
#               1. construct a batch of real samples from the training set, forward  pass through D, calculate the loss, calculate the gradients in a backward pass
#               2. construct a batch of fake samples with the current generator, forward pass this batch through D, calculate the loss, accumulate the gradients with a backward pass
#               3. finally, call a step of the discriminator's optimizer
#
# 2. Discriminator training
#       The goal is to maximize log(D(G(z)))
#       To do it: classify the generator output from Part 1 with the Discriminator, compute loss using real labels, compute gradients in a backward pass, and update the parameters with an optimizer step 
# 3. Finally, we will report the following statistics:
#       * Loss_D --> discriminator loss calculated as the sum of losses for the all real and fake batches (log(D(x))+log(D(G(z))))
#       * Loss_G --> generator loss calculated as log(D(G(z)))
#       * D(x) --> average output (across the batch) of the discriminator for the all real batch. This should start close to 1 then theoretically converge to 0.5 when G gets better (because D should become unable to distinguish fake and real images, so it will start constantly guessing with 50% confidence)
#       * D(G(z)) --> average discriminator outputs for the all fake batch. The first number is before D is updated and the second number is after D is updated. These numbers should start near 0 and converge to 0.5 as G gets better (same reason  as above)

img_list = []
G_losses = []
D_losses = []
iters = 0

print('Starting training loop...')

for epoch in range(num_epochs):
    # Iterate over the batches
    for i, data in enumerate(dataloader, 0):
        ##
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) #
        ##

        # Train with all-real batch #
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full(size=(b_size), fill_value=real_label, dtype=torch.float, device=device) # Creates a tensor of size "size" filled with values "fill_value"
        # Forward pass real batch through Discriminator
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for Discriminatorr in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train with all-fake batch #
        # Generate batch of latent vector
        noise = torch.randn(size=b_size, out=nz, 1, 1, device=device)
        # Generate fake image batch with Generator
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with Discriminator
        output = netD(
            fake.detach() # detach -> creates a tensor that share storage with tensor that does not require grad. It detaches the output from the computational graph
        ).view(-1)
        # Calculate loss on all-fake batch
        errD_fake = criterion(output, label)
        # Calculate gradients for Discriminatorr in backward pass
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update Discriminator
        optimizerD.step()

        ##
        # (2) Update G network: maximize log(D(G(z)))
        ##

        netG.zero_grad()
        label.fill_(real_label) # Fake labels are real for generator cost
        # Since we just updated Discriminator, perform another forward pass of all-fake batch through Discriminator
        output = netD(fake).view(-1)
        # Calculate Generator's loss
        errG = criterion(output, label)
        # Calculate gradients for Generator
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update Generator
        optimizerG.step()


        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        # Save losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving Generator's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1


# RESULTS #

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_lossees, label='G')
plt.plot(D_losses, label='D')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())


# Get a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
