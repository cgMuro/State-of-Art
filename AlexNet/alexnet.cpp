#include <iostream>
#include <torch/torch.h>

// ----------------------------------- MODEL ----------------------------------- //

struct AlexNetImplementation : torch::nn::Module
{
    AlexNetImplementation(int num_classes=1000)
    : 
        // Convolutions
        conv1(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, (11, 11)).stride(4).padding(2))),
        conv2(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, (5, 5)).stride(1).padding(2))),
        conv3(torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, (3, 3)).stride(1).padding(1))),
        conv4(torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, (3, 3)).stride(1).padding(1))),
        conv5(torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, (3, 3)).stride(1).padding(1))),
        // Maxpooling
        max_pool(torch::nn::MaxPool2d((3, 3), 2)),
        // Adaptive average pooling
        avgpool(torch::nn::AdaptiveAvgPool2d((6, 6))),
        // Linear
        linear(torch::nn::Linear(4096, 4096)),
        linear_out(torch::nn::Linear(4096, num_classes))
    {
        // Reegister all layers
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("conv5", conv5);
        register_module("max_pool", max_pool);
        register_module("avgpool", avgpool);
        register_module("linear", linear);
        register_module("linear_out", linear_out);
    }

    // Define forward function to pass the input through the network
    torch::Tensor forward(torch::Tensor x) 
    {
        // First Convolutional Layer + Maxpooling
        x = conv1(x);
        x = torch::relu(x);
        x = max_pool(x);
        // Second Convolutional Layer + Maxpooling
        x = conv2(x);
        x = torch::relu(x);
        x = max_pool(x);
        // Third Convolutional Layer + Maxpooling
        x = conv3(x);
        x = torch::relu(x);
        // Fourth Convolutional Layer + Maxpooling
        x = conv4(x);
        x = torch::relu(x);
        // Fifth Convolutional Layer + Maxpooling
        x = conv5(x);
        x = torch::relu(x);
        x = max_pool(x);
        // Adaptive average poolin
        x = avgpool(x);
        // Flatten the input
        x = torch::flatten(x, 1);
        // Fully connected network with dropout
        x = torch::dropout(x, 0.5, is_training());
        x = linear(x);
        x = torch::relu(x);
        x = torch::dropout(x, 0.5, is_training());
        x = linear(x);
        x = torch::relu(x);
        x = linear_out(x);

        return x;
    }

    // Module layers
    torch::nn::Conv2d conv1, conv2, conv3, conv4, conv5; 
    torch::nn::MaxPool2d max_pool;
    torch::nn::AdaptiveAvgPool2d avgpool;
    torch::nn::Linear linear, linear_out;
};
TORCH_MODULE_IMPL(AlexNet, AlexNetImplementation);   // Wraps AlexNetImplementation to AlexNet with a shared_ptr and abstracts away any memory management

int main() {
    // Device -> set CUDA if available
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        torch::Device device = torch::kCUDA;
        std::cout << "CUDA available" << std::endl;
    }

    // Init model
    auto alexnet = AlexNet();
    // Model on CUDA (if available, CPU otherwise)
    alexnet->to(device);

    // Optimizer
    torch::optim::Adam optimizer(alexnet->parameters(), torch::optim::AdamOptions(1e-3));

    // Dummy data
    int batch_size = 128;
    torch::Tensor x_train = torch::ones({batch_size, 3, 224, 224}, device);   // Input data
    torch::Tensor y_train = torch::randn({batch_size, 1000}, device);    // Target data



    // ----------------------------------- TRAINING ----------------------------------- //

    int EPOCHS = 1000;

    // Model in training mode
    alexnet->train();

    for (int i = 0; i < EPOCHS; i++) {
        // Gradients to zero
        optimizer.zero_grad();
        // Get predictions from model
        torch::Tensor output = alexnet->forward(x_train);
        // Calculate loss
        torch::Tensor loss = torch::mse_loss(output, y_train);
        // Calculate gradients
        loss.backward();
        // Update parameters
        optimizer.step();

        if (i % 10 == 0)
            std::cout << "Loss: " << loss << std::endl;
    }



    // ----------------------------------- TESTING ----------------------------------- //

    // Dummy data
    torch::Tensor x_test = torch::randn({batch_size, 3, 224, 224}, device);
    torch::Tensor y_test = torch::randn({batch_size, 1000}, device);

    // Model in validation mode
    alexnet->eval();

    // Get predictions from model
    torch::Tensor out_test = alexnet->forward(x_test);
    // Calculate the error
    torch::Tensor result = torch::mse_loss(out_test, y_test);

    std::cout << "Test result -> " << result << std::endl;
    // Log the predicted class
    std::cout << "Class -> " << torch::nn::functional::softmax(out_test, torch::nn::functional::SoftmaxFuncOptions(0));
}
