**WaveNet** is a probabilistic, autoregressive, deep learning model by DeepMind for generating raw audio waveforms.

It can be used for multi-speaker speech generation, text-to-speech tasks, generating novel music, and discriminative audio problems.


## Architecture

### Dilated Causal Convolution
<div align="center">
    <img src="https://benanne.github.io/images/wavenet.png" alt="Dilated Causal Convolution Image" width="600">
</div>

* It's the combination of causal and dilated convolutions  
* *Causal Convolution* -> a convolutional layer that respects the ordering of the data.
* *Dilated Convolution* -> a convolutional layer where the filter is applied over an area larger than its length by skipping input values with a certain step.

<br>

### Gated Activation Units
Applies the TanH and Sigmoid activation functions on the convolutions output and then performs element-wise multiplication over the results of the two functions.

<br>

### Residual and Skip Connections
Here's an overview of the entire architecture, along with the residual and skip connections.

<div align="center">
    <img src="https://i.stack.imgur.com/t7qkv.png" alt="Overview of the residual block and the entire architecture" width="600">
</div>

To speed up convergence and enable training of deeper models, both residual and parameterized skip connections are used throughout the network.

<br>

### Softmax distributions and Mu-law
* Softmax distributions are used to model the conditional distributions over individual audio samples.
* To make the process more tractable, a µ-law companding transformation is applied to the data.

<hr>
<br>

## Resources
* [*WaveNet: A Generative Model for Raw Audio*](https://arxiv.org/abs/1609.03499) (**Aaron van den Oord et al., 2016**)
* [WaveNet: A generative model for raw audio](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio) (**DeepMind Blog**)
* [WaveNet by Google DeepMind | Two Minute Papers #93](https://www.youtube.com/watch?v=CqFIVCD1WWo) (**Two Minute Papers**)
* [Mason-McGough/Wavenet-PyTorch](https://github.com/Mason-McGough/Wavenet-PyTorch) GitHub (**Mason-McGough**)
* [vincentherrmann/pytorch-wavenet](https://github.com/vincentherrmann/pytorch-wavenet) GitHub (**vincentherrmann**)