**BERT** (Bidirectional Encoder Representations from Transformers) is an **encoder-only transformer**.

To build BERT, we stack together multiple transformer encoders and apply 3 types of embeddings to the input.
Usually, BERT, it's first pre-trained on two unsupervised tasks (*masked language model* and *next sentence prediction*) to understand the language and then fine-tuned to solve a specific problem (e.g. classification, translation, etc).

* **Pre-training**:
    * **Embeddings** -> to handle non-numerical data and preserve token ordering it combines together 3 types of embeddings: **token embeddings** (vocabulary of pre-trained embeddings), **segment embeddings** (represents the sentence the token is part of), and **positional embeddings** (encodes the position of the token in the sentence)
    * **Masked Language Model (Mask LM)** -> words are randomly *masked* (i.e. removed, made invisible to the model) and BERT has to fill in the blank
    * **Next Sentence Prediction (NSP)** -> given two sentences BERT outputs the order of the sentences relative to one another (it's a binary value)
* **Fine-tuning**:
    * We take the pre-trained BERT and a dataset to perform the supervised task we want, add a final output linear layer (if needed) and then train the new model on the task.

<br>

<img src="https://miro.medium.com/max/1400/1*LtF3nUFDhP62e9XAs6SlyQ.png">

<br>

## Resources:
* Paper: [*"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"*](https://arxiv.org/abs/1810.04805) (**Jacob Devlin**, **Ming-Wei Chang**, **Kenton Lee**, **Kristina Toutanova**)
* [*BERT Neural Network - EXPLAINED!*](https://www.youtube.com/watch?v=xI0HHN5XKDo) (**CodeEmporium**)
* [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://www.youtube.com/watch?v=-9evrZnBorM) (**Yannic Kilcher**)
* [*Hugging Face Transformer: Training and fine-tuning*](https://huggingface.co/transformers/training.html) (**Hugging Face**)
* [*Illustrated BERT*](https://jalammar.github.io/illustrated-bert/) (**Jay Alammar**)
* [*A Visual Guide to Using BERT for the First Time*](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/) (**Jay Alammar**)