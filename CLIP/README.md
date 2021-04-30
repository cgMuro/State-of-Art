CLIP (Contrastive Language–Image Pre-training) is a model by OpenAI that connects images and text in a zero-shot fashion.

---

<p align='center'><b>What is contrastive learning?</b></p>
  It's a machine learning approach in which the model learns to distinguish objects by classifying them as similar or dissimilar.

<br>
<br>

<p align='center'><b>What does zero-shot mean?</b></p>
  Zero-shot transfer means having a model performing multiple different tasks from the one it was trained for without any fine-tuning.

---

What's remarkable about CLIP, is its ability to train on image-caption pairs found on the internet (which means it doesn't need a curated dataset), while still being able to accomplish, with good performances, a wide range of different tasks without fine-tuning.

To train CLIP by contrastive method, we have the model compare a batch of images with a batch of texts and learn to associate the pair that is most likely.
To do so, CLIP learns a multi-modal embedding space (images + texts embeddings) to maximize the cosine similarity of the embeddings and minimizing the cosine similarity of the incorrect pairs' embeddings.

<br>
<div align="center">
    <img src="https://openaiassets.blob.core.windows.net/$web/clip/draft/20210104b/overview-a.svg" alt="CLIP's contrastive pre-training" width="500">
</div>
<br>

### Architecture:
* **Image Encoder**  ->  either a *ResNet* with attention pooling or a *Vision Transformer*
* **Text Encoder**  ->  *Transformer*

### Pseudocode
From the paper at https://arxiv.org/pdf/2103.00020.pdf
```
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l] - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter
# extract feature representations of each modality

I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]

# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2
```

<br>


## Resources:
* Paper: [*"Learning Transferable Visual Models From Natural Language Supervision"*](https://arxiv.org/abs/2103.00020) (**Alec Radford**, **Jong Wook Kim**, **Chris Hallacy**, **Aditya Ramesh**, **Gabriel Goh**, **Sandhini Agarwal**, **Girish Sastry**, **Amanda Askell**, **Pamela Mishkin**, **Jack Clark**, **Gretchen Krueger**, **Ilya Sutskever**)
* [CLIP: Connecting Text and Images](https://openai.com/blog/clip/) (**OpenAI Blog**)
*  [OpenAI CLIP: ConnectingText and Images (Paper Explained)](https://www.youtube.com/watch?v=T9XSU0pKX2E) (**Yannic Kilcher**)
* [openai/CLIP GitHub](https://github.com/openai/CLIP) (**OpenAI**)
