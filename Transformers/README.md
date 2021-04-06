A **transformer** is an architecture that handles sequential data.    
It's based on the *attention mechanism*, which allows the algorithm to refer to previous sequences by learning to pay *"attention"* to some of them as needed.

The transformer is made of an encoder and a decoder.      
Both use:
* multi-head attention (a technique in which we calculate attention multiple times and average the result)
* feed-forward neural networks
* normalization (e.g. batch and layer normalization)
* residual connections
* embeddings

<br>

Here's how it works.

**Encoder**:
1. Takes in an input sequence and uses token embeddings to handle non-numerical data
2. Adds the positional embeddings (a type of embedding that encodes the position of a token in the sequence by calculating the distances between the tokens) to the token embeddings
3. Uses multi-head attention to calculate the importance that each token has with respect to the others
4. Then applies normalization and residual connections

**Decoder**:

5. Takes in the target tokens decoded up to the current step (i.e. the previous decoder's output or the token initiating the sequence) and process them as before (token embeddings + positional embeddings)
6. Then the sequence is masked (put simply: the future tokens that the algorithm doesn't have to see are made "invisible" by setting the embeddings to `-inf`) and put into a multi-head attention network
7. Then normalization and residual connections are applied
8. In the next phase, we pass the encoder's outputs and the masked multi-head attention, first into another multi-head attention layer and then into a feed-forward network

**Output**:

9. Finally, we have a feed-forward linear network and a softmax function that outputs the next token in the sequence

<br>
<div align="center">
    <img src="https://miro.medium.com/max/1090/1*HunNdlTmoPj8EKpl-jqvBA.png" alt="Transformer architecture image" width="500">
</div>

## The code
* `transformer.py` -> implementation of a simple transformer
* `seq2seq_transformer.py` -> implementation of a transformer handling a sequence to sequence problem ([source](https://pytorch.org/tutorials/beginner/transformer_tutorial.html))
* `annotated_transformer.py` -> walkthrough of the paper using code ([source](http://nlp.seas.harvard.edu/2018/04/03/attention.html
))
* `computer-vision` -> contains transformer applications in computer vision
    * `vision_transformer.py` -> implementation of the vision transformer ([source](https://github.com/lucidrains/vit-pytorch))

## Resources:
* [Paper: *"Attention Is All You Need"*](https://arxiv.org/abs/1706.03762) (**Ashish Vaswani**, **Noam Shazeer**, **Niki Parmar**, **Jakob Uszkoreit**, **Llion Jones**, **Aidan N. Gomez**, **Lukasz Kaiser**, **Illia Polosukhin**)
* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (**Jay Alammar**)
* [Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) (**Jay Alammar**)
* [Sequence-to-Sequence Modeling with nn.Transformer and TorchText](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) (**PyTorch**)
* [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) (**Guillaume Klein**, **Yoon Kim**, **Yuntian Deng**, **Jean Senellart**, **Alexander M. Rush**)
* [AI Language Models & Transformers](https://www.youtube.com/watch?v=rURRYI66E54) (**Computerphile**)
* [Transformer Neural Networks - EXPLAINED! (Attention is all you need)](https://www.youtube.com/watch?v=TQQlZhbC5ps) (**CodeEmporium**)
* [Attention Is All You Need](https://www.youtube.com/watch?v=iDulhoQ2pro) (**Yannic Kilcher**)
