#include <iostream>
#include <torch/torch.h>

// Define the structure for the BasicBlock -> 2 3x3 convolution and batch normalization + residual connection before the last relu 
struct BasicBlock : torch::nn::Module {

    int expansion = 1;

    BasicBlock(
        int in_planes,  // Block's input dimension
        int planes,     // Block's output dimension
        int stride=1,   // Convolutional layer's stride
        int groups=1,   // Convolutional layer's groups (-> controls the connections between inputs and outputs)
        int dilation=1  // Convolutional layer's dilation (-> spacing between the kernel points) 
    ) 
    :
        // First Convolution
        conv1(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes, (3, 3)).stride(stride).padding(dilation).groups(groups).dilation(dilation).bias(false))),
        // Second Convolution
        conv2(torch::nn::Conv2d(torch::nn::Conv2dOptions(planes, planes, (3, 3)).stride(stride).padding(dilation).groups(groups).dilation(dilation).bias(false))),
        // Batch Normalization
        batchNorm(torch::nn::BatchNorm2d(planes))
    {
        // Register layers
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("batchNorm", batchNorm);
    }

    torch::Tensor forward(torch::Tensor x) {
        // Define tensor "out"
        torch::Tensor out;

        // Pass the input throught the first layer
        out = conv1(x);
        out = batchNorm(out);
        out = torch::relu(out);
        // Pass throught the second layer
        out = conv2(out);
        out = batchNorm(out);
        out += x;  // Residual connection
        out = torch::relu(out);

        return out;
    }


    // Define layers
    torch::nn::Conv2d conv1, conv2;
    torch::nn::BatchNorm2d batchNorm;
};


// Define the structure for the Bottleneck, which is used to reduce computation. Architecture:
//      1. 1x1 convolution -> which scales down the vector's dimension
//      2. 3x3 convolution
//      3. 1x1 convolution -> which scales up the vector's dimension to its original size
struct Bottleneck : torch::nn::Module {

    int expansion = 4;

    Bottleneck(
        int in_planes,  // Block's input dimension
        int planes,     // Block's output dimension
        int stride=1,   // Convolutional layer's stride
        int groups=1,   // Convolutional layer's groups (-> controls the connections between inputs and outputs)
        int dilation=1  // Convolutional layer's dilation (-> spacing between the kernel points)
    )
    :
        // First Convolution
        conv1(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes*groups, (1, 1)).stride(stride).bias(false))),
        // Second Convolution
        conv2(torch::nn::Conv2d(torch::nn::Conv2dOptions(planes*groups, planes*groups, (3, 3)).stride(stride).padding(dilation).groups(groups).dilation(dilation).bias(false))),
        // Third Convolution
        conv3(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes*expansion, (1, 1)).stride(stride).bias(false))),
        // First Batch Normalization
        batchNorm1(torch::nn::BatchNorm2d(planes*groups)),
        // Second Batch Normalization
        batchNorm2(torch::nn::BatchNorm2d(planes*expansion))
    {
        // Register layers
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("batchNorm1", batchNorm1);
        register_module("batchNorm2", batchNorm2);
    }

    torch::Tensor forward(torch::Tensor x) {
        // Define tensor "out"
        torch::Tensor out;

        // Pass throught the 1x1 conv (scales down)
        out = conv1(x);
        out = batchNorm1(out);
        out = torch::relu(out);

        // Pass throught the 3x3 conv
        out = conv2(out);
        out = batchNorm1(out);
        out = torch::relu(out);

        // Pass throught the 1x1 conv (scales up)
        out = conv3(out);
        out = batchNorm2(out);
        out += x;  // Residual connection
        out = torch::relu(out);

        return out;
    }

    // Define layers
    torch::nn::Conv2d conv1, conv2, conv3;
    torch::nn::BatchNorm2d batchNorm1, batchNorm2;
};


struct ResNet : torch::nn::Module {

    int in_planes = 64;

    ResNet(
        bool useBottleneck,             // What type of block to use. We use bottlenecks if true otherwise basic blocks
        int layers[4],                  // Number of blocks to use for each "layer"
        int num_classes=1000,           // Number of classes to classify
        bool zero_init_residual=false,  // If false we don't initiate the weights of the blocks, otherwise we do
        int groups=1                    // Convolutional layer's groups (-> controls the connections between inputs and outputs) 
    )
    :
        // Convolution
        conv(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, in_planes, (7, 7)).stride(2).padding(3).bias(false))),
        // Batch Normalization
        batchNorm(torch::nn::BatchNorm2d(in_planes)),
        // Maxpooling
        maxPool(torch::nn::MaxPool2d(3, 2, 1)),
        // BasicBlock or Bottleneck layers
        layer1(_make_layer(true, 64, layers[0])),
        layer2(_make_layer(true, 128, layers[1])),
        layer3(_make_layer(true, 256, layers[2])),
        layer4(_make_layer(true, 512, layers[3])),
        // Adaptive Average Pooling
        avgPool(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}))),
        // Feed Forward Linear Layer
        fc(torch::nn::Linear(512*calc_epansion(useBottleneck), num_classes))
    {
        // Register layers
        register_module("conv", conv);
        register_module("batchNorm", batchNorm);
        register_module("maxPool", maxPool);
        register_module("avgPool", avgPool);
        register_module("fc", fc);

        // Initialize the weights
        for (auto &m: this->modules()) {
            initialize_weights(*m, zero_init_residual);
        }
    }

    // Function to calculate the number of dimensions (which is based on the type of block we are using)
    int calc_epansion(bool useBottleneck) {
        if (useBottleneck) {
            return 4;
        } else {
            return 1;
        }
    };

    // Function to initialize the weights based on the type of layer
    void initialize_weights(torch::nn::Module& module, bool zero_init_residual) {
        if (auto* m = module.as<torch::nn::Conv2d>()) {
            // Fills the input Tensor with values according to a normal distribution
            torch::nn::init::kaiming_normal_(m->weight, (0.0), torch::kFanOut, torch::kReLU);
        } else if (auto* m = module.as<torch::nn::BatchNorm2d>()) {
            // Fills the input Tensor with the value "val"
            torch::nn::init::constant_(m->weight, 1);
            torch::nn::init::constant_(m->bias, 0);
        }

        if (zero_init_residual) {
            // Check if the architecture being used is Bottleneck
            if (auto* m = module.as<Bottleneck>()) {
                torch::nn::init::constant_(m->batchNorm2->weight, 0);
            // Check if the architecture being used is BasicBlock
            } else if (auto* m = module.as<BasicBlock>()) {
                torch::nn::init::constant_(m->batchNorm->weight, 0);
            };
        }
    }

    // Function to create the layers based on the type of block passed
    torch::nn::SequentialImpl _make_layer(bool useBottleneck, int planes, int n_blocks=4, int stride=1, int groups=1, bool dilate=false) {
        // Init dilation
        int dilation = 1;

        // Define block
        BasicBlock block = BasicBlock(in_planes, planes, stride=1, groups=groups, dilation=1);
        if (useBottleneck) {
            Bottleneck block = Bottleneck(in_planes, planes, stride=1, groups=groups, dilation=1);
        };
        // Check if dilate is true (-> which means if we want to increasee the dialtion)
        if (dilate) {
            dilation *= stride;
            stride = 1;
        };

        // Init and build the layers
        torch::nn::SequentialImpl layers = torch::nn::SequentialImpl();

        for (int i=0; i < sizeof(n_blocks); i++) {
            layers.push_back(block);
        };

        return layers;
    }

    torch::Tensor forward(torch::Tensor x) {
        // Pass input into the first block
        x = conv(x);
        x = batchNorm(x);
        x = torch::relu(x);
        x = maxPool(x);
        // Pass input through the residual blocks (either BasicBlocks or Bottlenecks)
        x = ResNet::layer1.forward(x);
        x = ResNet::layer2.forward(x);
        x = ResNet::layer3.forward(x);
        x = ResNet::layer4.forward(x);
        // Fully connected layers for classification preceded by an Average pooling layer
        x = avgPool(x);
        x = torch::flatten(x, 1);
        x = fc(x);
        
        return x;
    }

    // Define layers
    torch::nn::Conv2d conv;
    torch::nn::BatchNorm2d batchNorm;
    torch::nn::MaxPool2d maxPool;
    torch::nn::AdaptiveAvgPool2d avgPool;
    torch::nn::Linear fc;
    torch::nn::SequentialImpl layer1, layer2, layer3, layer4;
};
