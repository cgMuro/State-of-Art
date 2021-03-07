// https://github.com/pytorch/examples/tree/master/cpp/dcgan

#include <iostream>
#include <cmath>
#include <cstdio>
#include <torch/torch.h>

// The size of the noise vector fed to the generator.
const int64_t kNoiseSize = 100;

// The batch size for training.
const int64_t kBatchSize = 64;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 30;

// Where to find the MNIST dataset.
const char *kDataFolder = "./data";

// After how many batches to create a new checkpoint periodically.
const int64_t kCheckpointEvery = 200;

// How many images to sample at every checkpoint.
const int64_t kNumberOfSamplesPerCheckpoint = 10;

// Set to `true` to restore models and optimizers from previously saved
// checkpoints.
const bool kRestoreFromCheckpoint = false;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

using namespace torch;

// DCGAN Network (Discriminator + Generator)
struct DCGANGeneratorImpl : nn::Module
{
    DCGANGeneratorImpl(int kNoiseSize)
        : conv1(nn::ConvTranspose2dOptions(kNoiseSize, 256, 4).bias(false)),
          batch_norm1(256),
          conv2(nn::ConvTranspose2dOptions(256, 128, 3)
                    .stride(2)
                    .padding(1)
                    .bias(false)),
          batch_norm2(128),
          conv3(nn::ConvTranspose2dOptions(128, 64, 4)
                    .stride(2)
                    .padding(1)
                    .bias(false)),
          batch_norm3(64),
          conv4(nn::ConvTranspose2dOptions(64, 1, 4)
                    .stride(2)
                    .padding(1)
                    .bias(false))
    {
        // we need register_module() to use the parameters later on
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("batch_norm1", batch_norm1);
        register_module("batch_norm2", batch_norm2);
        register_module("batch_norm3", batch_norm3);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(batch_norm1(conv1(x)));
        x = torch::relu(batch_norm2(conv2(x)));
        x = torch::relu(batch_norm3(conv3(x)));
        x = torch::tanh(conv4(x));
        return x;
    }

    nn::ConvTranspose2d conv1, conv2, conv3, conv4;
    nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};
TORCH_MODULE(DCGANGenerator);

int main()
{
    torch::manual_seed(1);

    // GPU settings if available
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA id available! Training on GPU." << std::endl;
        device = torch::kCUDA;
    }

    DCGANGenerator generator(kNoiseSize);
    generator->to(device);

    // Discriminator
    nn::Sequential discriminator(
        // First layer
        nn::Conv2d(nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // Second layer
        nn::Conv2d(nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(128),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // Third layer
        nn::Conv2d(nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(256),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // Fourth layer
        nn::Conv2d(nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
        nn::Sigmoid());
    discriminator->to(device);

    // ------------- DATA ------------- //

    // Create dataset
    auto dataset = torch::data::datasets::MNIST(kDataFolder)
                       .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                       .map(torch::data::transforms::Stack<>());

    const int64_t batches_per_epoch = std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));

    // Create dataloader
    auto data_loader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

    // Load batches of data
    for (torch::data::Example<> &batch : *data_loader)
    {
        std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
        for (int64_t i = 0; i < batch.data.size(0); i++)
        {
            std::cout << batch.target[i].item<int64_t>() << " ";
        }
        std::cout << std::endl;
    }

    // ------------- TRAINING ------------- //

    // Optimizers
    torch::optim::Adam generator_optimizer(
        generator->parameters(), torch::optim::AdamOptions(2e-4).betas(std::make_tuple (0.5, 0.5))
    );
    torch::optim::Adam discriminator_optimizer(
        discriminator->parameters(), torch::optim::AdamOptions(5e-4).betas(std::make_tuple (0.5, 0.5))
    );

    // Restore saved checkpoints
    if (kRestoreFromCheckpoint)
    {
        torch::load(generator, "generator-checkpoint.pt");
        torch::load(generator_optimizer, "generator-optimizer-checkpoint.pt");
        torch::load(discriminator, "discriminator-checkpoint.pt");
        torch::load(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
    }

    // Training loop
    int64_t checkpoint_counter = 1;
    for (int64_t epoch = 1; epoch < kNumberOfEpochs; epoch++)
    {
        int64_t batch_index = 0;
        for (torch::data::Example<> &batch : *data_loader)
        {
            // Train discriminator with real images
            discriminator->zero_grad();
            torch::Tensor real_images = batch.data.to(device);
            torch::Tensor real_labels = torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
            torch::Tensor real_output = discriminator->forward(real_images);
            torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
            // // Calculate the gradient for each parameter
            d_loss_real.backward();

            // Train discriminator with fake images
            torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1});
            torch::Tensor fake_images = generator->forward(noise);
            torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);
            torch::Tensor fake_output = discriminator->forward(fake_images.detach());
            torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
            // Calculate the gradient for each parameter
            d_loss_fake.backward();

            // Total loss
            torch::Tensor d_loss = d_loss_real + d_loss_fake;
            // Update parameters
            discriminator_optimizer.step();

            // Train generatorr
            generator->zero_grad();
            fake_labels.fill_(1);
            fake_output = discriminator->forward(fake_images);
            torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
            g_loss.backward();
            generator_optimizer.step();

            std::printf(
                "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f",
                epoch,
                kNumberOfEpochs,
                ++batch_index,
                batches_per_epoch,
                d_loss.item<float>(),
                g_loss.item<float>());

            // Checkpoint model and optimizer state
            if (batch_index % kCheckpointEvery == 0)
            {
                torch::save(generator, "generator-checkpoint.pt");
                torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
                torch::save(discriminator, "discriminator-checkpoint.pt");
                torch::save(discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
                // Sample the generator and save the images
                torch::Tensor samples = generator->forward(torch::randn({8, kNoiseSize, 1, 1}, device));
                torch::save((samples + 1.0) / 2.0, torch::str("dcgan-sample-", checkpoint_counter, ".pt"));
                std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
            }
        }
    }

    std::cout << "Training complete!" << std::endl;
}
