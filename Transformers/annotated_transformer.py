# http://nlp.seas.harvard.edu/2018/04/03/attention.html

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

seaborn.set_context(context='talk')


# ----------------------------------- Model Architecture ----------------------------------- #

# The model is composed by an Encoder and a Decoder.
# The Encoder maps an input sequence (x_1, ..., x_n) to a sequence of continuous representations z = (z_1, ..., z_n).
# The decoder then, given z, generates an output sequence (y_1, ..., y_n) one step at a time.
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences"
        return self.decode(
            self.encode(src, src_mask), src_mask, tgt, tgt_mask
        )

    # Define encoding
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    # Define decoding
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step"
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# ----------------------------------- Encoder and Decoder Stacks ----------------------------------- #

# ENCODER #

# The encoder is composed of a stack of 6 identical layers
def clones(module, N):
    "Produce N identical layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# Define normalization
class LayerNorm(nn.Module):
    "Construct a layernorm module"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    "A residual connection followed by a layer norm."
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size"
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attention and feed forward network"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size=size, dropout=dropout), 2)
        self.sizee = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# DECODER #

# The decoder is composed of a stack of 6 identical layers
class Decoder(nn.Module):
    "Generic N layer decoder with masking"
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attention, src-attn, and feed forward"
    def __init__(self, size, self_attn, src_att, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_att
        self.feed_forward = feed_forward
        # Define 3 sublayers
        self.sublayer = clones(SublayerConnection(size=size, dropout=dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.self_attn(x, x, x, src_mask)) # performs multi-head attention over the output of the encoder stack
        return self.sublayer[2](x, self.feed_forward)


# Modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. 
# This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i 
# can depend only on the known outputs at positions less than i.
def subsequent_mask(size):
    "Mask out subsequent positions"
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# ATTENTION #

# An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and 
# output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed 
# by a compatibility function of the query with the corresponding key
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaleed Dot Product Attention'"
    # Number of dimensions
    d_k = query.size(-1)
    # Calcualte score by multiplying query and key vectors and then normalizing by dividing it by the sqaure root of the number of dimensions
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Mask the scores for the decoder
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # Get the softmax of the scores to highlight the bigger ones and reduce the influece of the smaller ones
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        # Apply dropout
        p_attn = dropout(p_attn)
    
    # Return the z vector (value vector multiplied by its weight) and the softmaxed scores
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads"
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)
    
        # 1) Do all the linear projections in batch from d_model => h * d_k
        query, key, value = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query=query, key=key, value=value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)

        return self.linears[-1](x)


# POSITION-WISE FEED-FORWARD NETWORKS #
# Each of the layers in the encoder and decoder contains a fully connected feed-forward network, consisting of two linear transformations with a ReLU activation in between
# FFN(x) = max(0, xW_1 + b_1) * W_2 + b_2
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# EMBEDDINGS AND SOFTMAX #
# We use learned embeddings to convert the input tokens and output tokens to vectors of dimension "d_model"
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# POSITIONAL ENCODING #
# We use positional encoding to give the model a sense of what is the absolute and relative possition of the current word.
# To do it we calculate the positional encoding vector (which has d_model dimensions) and we sum it with the encoding vector of the word.
# To calculate the positional encoding we use the sine and cosine functions of different frequencies -> each dimension of the positional encoding corresponds to a sinusoid
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

# FULL MODEL
def make_model(
    src_vocab,    # Input examples vocabulary
    tgt_vocab,    # Target data vocabulary
    N=6,          # Number of encoder layers
    d_model=512,  # Number of embedding dimensions
    d_ff=2048,    # Number of features
    h=8,          # Number of heads in MultiHeadedAttention
    dropout=0.1   # Dropout rate
):
    "Function that takes in all parameters and produces the full model"
    # Define deep copy -> which constructs a new compound object and then, recursively, inserts copies into it of the objects found in the original
    c = copy.deepcopy
    # Multi-head Attention
    attn = MultiHeadedAttention(h=h, d_model=d_model, dropout=dropout)
    # Position Wise Feed Forward Network
    ff = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
    # Positional Encoding
    position = PositionalEncoding(d_model=d_model, dropout=dropout)
    # Encode-Decoder model
    model = EncoderDecoder(
        # Encoder
        Encoder(EncoderLayer(size=d_model, self_attn=c(attn), feed_forward=c(ff), dropout=dropout), N=N),
        # Decodeer
        Decoder(DecoderLayer(size=d_model, self_attn=c(attn), src_att=c(attn), feed_forward=c(ff), dropout=dropout), N=N),
        # Final feed forward network
        nn.Sequential(Embeddings(d_model=d_model, vocab=src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model=d_model, vocab=tgt_vocab), c(position)),
        Generator(d_model=d_model, vocab=tgt_vocab)
    )

    # Initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model



# ----------------------------------- Encoder and Decoder Stacks ----------------------------------- #


# BATCHES AND MASKING #
# Define a batch object that holds the src and target sentences for training, as well as constructing the masks.
class Batch:
    "Object for holding a batch of data with mask during training"
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

# TRAINING LOOP #
# Create a training and scoring function to keep track of loss. We also pass in a loss compute function that handles parameter updates.
def run_epoch(data_iter, model, loss_compute):
    "Traning and logging function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens

# TRAINING DATA AND BATCHING #
global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch

    return max(src_elements, tgt_elements)

# OPTIMIZER #
# We use the Adam optimizer and vary the learning rate over the course of training: it's increased linearly for the first warmup 
# training steps, and decreased thereafter proportionally to the inverse square root of the step number.
class NoamOpt:
    "Optim wrapper that implements rate"
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(
        model_size=model.src_embed[0].d_model, factor=2, warmup=4000, optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )

# REGULARIZATION #
# Label smoothing implementation using Kullback-Leibler divergence loss. 
# Instead of using a one-hot target distribution, we create a distribution that has confidence of the correct word and the rest of the smoothing mass distributed throughout the vocabulary.
class LabelSmoothing(nn.Module):
    "Implement label smoothing"
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

crit = LabelSmoothing(5, 0, 0.1)
def loss(x):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(Variable(predict.log()), Variable(torch.LongTensor([1]))).data[0]



# ----------------------------------- TRAINING EXAMPLE: SYNTHETIC DATA ----------------------------------- #

# SYNTHETIC DATA #
def data_gen(V, batch, n_batches):
    "Generate random data for a src-tgt copy task"
    for i in range(n_batches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)

# LOSS COMPUTATION #
class SimpleLossCompute:
    "Loss compute and train function"
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
    
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data[0] * norm

# GREEDY DECODING #
# In greedy decoding we straightforward choose the word that has the highest probability (given by the softmax) as the output.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model_size=model.src_embed[0].d_model, factor=1, warmup=400, optimizer=torch.optim.Adam(model.parameters(), lr=0,  betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
    model.train()
    run_epoch(data_gen(V=V, batch=30, n_batches=20), model, SimpleLossCompute(generator=model.generator, criterion=criterion, opt=model_opt))
    model.eval()
    print(run_epoch(data_gen(V=V, batch=30, n_batches=5), model, SimpleLossCompute(generator=model.generator, criterion=criterion, opt=None)))

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generato(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]))
src_mask = Variable(torch.ones(1, 1, 10))
print(greedy_decode(model=model, src=src, src_mask=src_mask, max_len=10, start_symbol=1))
