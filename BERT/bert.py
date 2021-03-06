# https://github.com/dhlee347/pytorchic-bert

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# Utility functions
def split_last(x, shape):
    "Split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "Merge last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


# ----------------------------------- BERT MODEL ----------------------------------- #

# Configurations
vocab_size : int = None      # Size of vocabulary
dim : int = 768              # Dimension of hidden layer in transformer encoder
n_layers : int = 12          # Number of hidden layers
n_heads : int = 12           # Number of heads in multi-headed attention layers
n_units : int = 768*4        # Dimension of intermidiate layers in positionwise feed forward net
dropout_hidden : float = 0.1 # Dropout rate in hidden layers
dropout_attn : float = 0.1   # Dropout rate in attention layers
max_len : int = 512          # Maximum length for positional encoding
n_segments : int = 2         # Number of sentence segments


class LayerNorm(nn.Module):
    "Definition of a normalization layer"
    def __init__(self, dim, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma + x + self.beta

class Embeddings(nn.Module):
    "The embedding module for word, position and token_type embeddings."
    def __init__(self, vocab_size, dim, n_segments, dropout_hidden):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)    # Token embedding
        self.position_embed = nn.Embedding(max_len, dim)    # Position embedding
        self.seg_embed = nn.Embedding(n_segments, dim)      # Segment (token type) embedding

        self.norm = LayerNorm(dim=dim)
        self.drop = nn.Dropout(dropout_hidden)
    
    def forward(self, x, segment):
        sequence_len = x.size(1)
        position = torch.arange(sequence_len, dtype=torch.long, device=x.device)
        position = position.unsqueeze(0).expand_as(x)

        embeddings = self.token_embed(x) + self.position_embed(position) + self.seg_embed(segment)
        return self.drop(self.norm(embeddings))


class MultiHeadedSelfAttention(nn.Module):
    "Multi-headed dot product attention"
    def __init__(self, dim, n_heads, dropout_attn):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout_attn)
        self.scores = None
        self.n_heads = n_heads

    def forward(self, x, mask):
        '''
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        '''
        # Get the 3 vectors q (query), k (key), v (value)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])

        # Calculate scores and normalize them
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transposee(-2, -1) / np.sqrt(k.size(-1))

        # Apply mask if required
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)

        # Extract the weight of each score using softmax
        scores = self.drop(F.softmax(scores, dim=-1))

        # Multiply each score with the correspoding v vector -> to get the actual a vector representation of the importance of each token
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        z = (scores @ v).transpose(1, 2).contiguous()

        # -merge-> (B, S, D)
        z = merge_last(z, 2)
        self.scores = scores

        return z

class PositionWiseFeedForward(nn.Module):
    "FeedForward Neural Network for each position"
    def __init__(self, dim, n_units):
        super().__init__()
        self.fc1 = nn.Linear(in_features=dim, out_features=n_units)
        self.fc2 = nn.Linear(in_features=n_units, out_features=dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):
    "Transformer Block"
    def __init__(self, dim, n_heads, n_units, dropout_hidden, dropout_attn):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, dropout_attn, n_heads)
        self.proj = nn.Linear(dim, dim)
        self.norm = LayerNorm(dim)
        self.pwff = PositionWiseFeedForward(dim, n_units)
        self.drop = nn.Dropout(dropout_hidden)
    
    def forward(self, x, mask):
        h = self.attn(x, mask)                      # Multi-headed attention
        h = self.norm(x + self.drop(self.proj(h)))  # Linear net + residual connection + normalization
        h = self.norm(h + self.drop(self.pwff(h)))  # Position wise net + residucal connection + normalization
        return h

class Transformer(nn.Module):
    "Transformer with self-attentive blocks"
    def __init__(self, vocab_size, dim, n_heads, n_layers, n_units, n_segments, dropout_hidden, dropout_attn):
        super().__init__()
        # Define embeddings
        self.embedding = Embeddings(vocab_size, dim, n_segments, dropout_hidden)
        # Define list containing multiple transformer encoding blocks
        self.blocks = nn.ModuleList(
            [Block(dim, dropout_attn, n_heads, n_units, dropout_hidden) for _ in range(n_layers)]
        )

    def forward(self, x, segment, mask):
        h = self.embed(x, segment)
        for block in self.blocks:
            h = block(h, mask)


# ----------------------------------- TRAINING ----------------------------------- #


# Hyperparameters for training
seed: int = 3431          # Random seed
batch_size: int = 32      # Batch size
lr: int = 5e-5            # Learning rate
n_epochs: int = 10        # Number of epochs
warmup: float = 0.1       # Linearly increase learning rate up to this point
save_steps: int = 100     # Interval for saving model
total_steps: int = 100000 # Total number of steps to train


class Trainer(object):
    "Training Helper Class"
    def __init__(self, batch_size, n_epochs, save_steps, total_steps, model, data_iter, optimizer, save_dir, device):
        self.model = model
        self.data_iter = data_iter # iterator to load data
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.device = device

    def train(self, get_loss, model_file=None, pretrain_file=None, data_parallel=True):
        "Training Loop"
        # Model in training mode
        self.model.train()
        # Load model
        self.load(model_file, pretrain_file)
        # Move model to CUDA
        model = self.model.to(self.device)
        # Use Data Parallelism with Multi-GPU
        if data_parallel:
            model = nn.DataParallel(model)

        # Global iteration steps regardless of epochs
        global_step = 0
        
        for e in range(n_epochs):
            # Sum of iteration losses to get average loss in every epoch
            loss_sum = 0.
            iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]

                self.optimizer.zero_grad()
                loss = get_loss(model, batch, global_step).mean() # mean() for data parallelism
                # Calculate gradients
                loss.backward()
                # Update parameters
                self.optimizer.step()

                # Update global step
                global_step += 1
                # Update total loss
                loss_sum += loss.item()
                iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())

                if global_step % save_steps == 0: # save
                    self.save(global_step)

                if total_steps and total_steps < global_step:
                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, n_epochs, loss_sum/(i+1)))
                    print('The Total Steps have been reached.')
                    self.save(global_step) # Save and finish when global_steps reach total_steps
                    return
            print('Epoch %d/%d : Average Loss %5.3f'%(e+1, n_epochs, loss_sum/(i+1)))

        self.save(global_step)

    def eval(self, evaluate, model_file, data_parallel=True):
        "Evaluation Loop"
        # Model in evaluation mode
        self.model.eval()
        # Load model
        self.load(model_file, None)
        # Move model to CUDA
        model = self.model.to(self.device)
        # Use Data Parallelism with Multi-GPU
        if data_parallel:
            model = nn.DataParallel(model)

        # Prediction results
        results = []
        iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')

        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]
            # Evaluation without gradient calculation
            with torch.no_grad():
                # Get model predictions
                accuracy, result = evaluate(model, batch)
            results.append(result)

            iter_bar.set_description('Iter(acc=%5.3f)'%accuracy)
        return results

    def load(self, model_file, pretrain_file):
        "Load saved model or pretrained transformer (a part of model)"
        if model_file:
            print('Loading the model from', model_file)
            self.model.load_state_dict(torch.load(model_file))
        elif pretrain_file: # Use pretrained transformer
            print('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.pt'): # Pretrain model file in PyTorch
                # Load only transformer parts
                self.model.transformer.load_state_dict(
                    {
                        key[12:]: value for key, value in torch.load(pretrain_file).items() if key.startswith('transformer')
                    }
                )

    def save(self, i):
        "Save current model"
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))
