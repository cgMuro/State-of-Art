# https://github.com/karpathy/minGPT

"""
Model Architecture:
    1. Combination of token encoding and positional encoding
    2. Uniform sequence of Transformer blocks:
        * each transformer = self-attention block + feed forward neural network 
        * all blocks use residual connections
    3. Linear projections into vanilla Softmax classifier
"""

import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)



# ----------------------------------- MODEL ----------------------------------- #

class GPTConfig:
    """ Base GPT configuration. These params are common to all GPT versions """
    embeddings_dropout = 0.1
    residual_dropout = 0.1
    attention_dropout = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size

        # Set other attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network (roughly 125M parameters) """
    n_layers = 12
    n_heads = 12    
    n_embeddings = 768

class GPT2Config(GPTConfig):
    """ GPT-2 like network (roughly 1.3B parameters) """
    n_layers = 24
    n_heads = 24    
    n_embeddings = 2048

class GPT3Config(GPTConfig):
    """ GPT-3 like network (roughly 175B parameters) """
    n_layers = 96
    n_heads = 96    
    n_embeddings = 12288


class CausalSelfAttention(nn.Module):
    """ A vanilla multi-head masked self-attention layer with a projection at the end """

    def __init__(self, config):
        super().__init__()
        assert config.n_embeddings % config.n_heads == 0  # Check if true otherwise raise assertion error

        # Key, query, value projections for all heads -> we use a linear network to extrapolate the three vectors (key, query, value) from the input vector
        self.key = nn.Linear(config.n_embeddings, config.n_embeddings)
        self.query = nn.Linear(config.n_embeddings, config.n_embeddings)
        self.value = nn.Linear(config.n_embeddings, config.n_embeddings)

        # Regularization
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

        # Output projection
        self.projection = nn.Linear(config.n_embeddings, config.n_embeddings)

        # Casual mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            'mask', 
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        )

        self.n_heads = config.n_heads

    def forward(self, x, layer_past=None):
        # Extrapolate dimensions from batch of input vectors
        B, T, C = x.size()

        # Calculate query, key, value vectors for all heads in batch and move head forward to be the batch dimension
        k = self.key(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)   # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        # Casual self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))        # Get scores and normalize them
        attention = attention.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))  # Mask "future" tokens
        attention = F.softmax(attention, dim=-1)                                     # Softmax the scores
        attention = self.attention_dropout(attention)                                # Apply dropout
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = attention @ v                                                            # Scale each value vector by its corresponding score
        y = y.transpose(1, 2).contiguous().view(B, T, C)                             # Put together the outputs from all heads

        # Output projection
        y = self.residual_dropout(self.projection(y))                                # Use a linear network to best concatenate the results from attention and apply dropout
        
        return y


class Block(nn.Module):
    """ An unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        # Define Normalization Layers -> which apply layer normalization over a mini-batch of input
        self.ln1 = nn.LayerNorm(normalized_shape=config.n_embeddings)
        self.ln2 = nn.LayerNorm(config.n_embeddings)
        # Define multi-head masked self-attention layer
        self.attention = CausalSelfAttention(config)
        # Define feed forward neural network (MultiLayer Perceptron)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embeddings, 4 * config.n_embeddings),  # The first layer projects the input to 4 times the model dimension (embeddings size) -> because so far is the size that was found to work best with the transformer
            nn.GELU(),                                                # Apply Gaussian Error Linear Units (GELU) activation function
            nn.Linear(4 * config.n_embeddings, config.n_embeddings),  # The second layer projects the input back to its origina size
            nn.Dropout(config.residual_dropout)                       # Apply dropout
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))  # Normalize -> self-attention -> residual connection
        x = x + self.mlp(self.ln2(x))        # Normalize -> multilayer perceptron -> residual connection
        return x

class GPT(nn.Module):
    """ The full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # Define the system to embed the input -> token embeddings + positional encoding
        self.token_embeddings = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.n_embeddings)
        self.positional_embeddings = nn.Parameter(data=torch.zeros(1, config.block_size, config.n_embeddings))
        self.dropout = nn.Dropout(config.embeddings_dropout)
        # Transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])  # Stack together "n_layers" of transformers
        # Decoder head -> the final part of the model which decodes the output of the transformer and returns the resulting token
        self.ln_f = nn.LayerNorm(config.n_embeddings)
        self.head = nn.Linear(config.n_embeddings, config.vocab_size, bias=False)

        self.block_size = config.block_size
        # Apply custom weight initialization
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        # Check if the module passed is either a linear layer, an embedding layer or both
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)  # Random Normal strategy -> initialization to values sample from the normal distribution
            # Check if the module passed is a linear layer
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()   # Zero the bias
        # Check if the module passed is a normalization layer
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()        # Zero the bias
            module.weight.data.fill_(1.0)   # Initialize the parameters to 1.0

    def configure_optimizers(self, train_config):
        """
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # Separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # Get layers that will experience regularizing weight decay
        whitelist_weight_modules = (torch.nn.Linear, )
        # Get layers that will NOT experience regularizing weight decay
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        # Iterate through the modules
        for mn, m in self.named_modules():  # mn = name of the module | m = module itself
            # Iterate through module parameters
            for pn, p in m.named_parameters():  # pn = name of the parameter | p = parameter itself.
                fpn = '%s.%s' % (mn, pn) if mn else pn  # Full parameter name

                # Check if it's a bias
                if pn.endswith('bias'):
                    no_decay.add(fpn)  # All biases will not be decayed

                # Check if it's a weight and it's in the list of layers that experience regularizing weight decay
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)     # Weights of whitelist modules will be weight decayed

                # Check if it's a weight and it's in the list of layers that DON'T experience regularizing weight decay
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)  # Weights of blacklist modules will NOT be weight decayed


        # Special case: the positional embedding parameter in the root GPT module as not decayed
        no_decay.add('positional_embeddings')

        # Validate that we considered every parameter
        param_dict = { pn: p for pn, p in self.named_parameters() }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, 'parameters %s made it into both decay/no_decay sets' % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, 'parameters %s were not separated into either decay/no_decay set' % (str(param_dict.keys() - union_params), )

        # Create the PyTorch optimizer object
        optim_groups = [
            { 'params': [param_dict[pn] for pn in sorted(list(decay))], 'weight_decay': train_config.weight_decay },
            { 'params': [param_dict[pn] for pn in sorted(list(no_decay))], 'weight_decay': 0.0 }
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        # Get input size
        b, t = idx.size()
        assert t <= self.block_size, 'Cannot forward, model block size is exhausted'

        # Forward the GPT model
        token_embeddings = self.token_embeddings(idx)                 # Each index maps to a (learnable) vector
        positional_embeddings = self.positional_embeddings[:, :t, :]  # Each position maps to a (learnable) vector
        x = self.dropout(token_embeddings + positional_embeddings)    # Apply dropout
        x = self.blocks(x)                                            # Pass input into the sequential container of transformers
        x = self.ln_f(x)                                              # Normalization in the input decoder
        logits = self.head(x)                                         # Pass input into the final network that returns the result in vocab size dimension

        # If we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss



# ----------------------------------- TRAINER ----------------------------------- #

class TrainerConfig:
    # Optimization parameters
    max_epochs = 10         # Number of epochs
    batch_size = 64         # Size of the mini batches
    learning_rate = 3e-4    # Learning rate value
    betas = (0.9, 0.95)     # Coefficients used for computing running averages of gradient and its square
    grad_norm_clip = 1.0    # Max normalization value of the gradients for gradient clipping
    weight_decay = 0.1      # Only  applied on matrix multiplication weights
    # Learning rate decay params -> linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6   # The learning rate will increase from 0 to the specified value until it hits the number of warmup tokens
    final_tokens = 260e9    # At what point we reach 10% of original learning rate
    # Checkpoint  settings
    checkpoint_path = None
    num_workers = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # CUDA settings
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            # Implements data parallelism at the module level by splitting the input and replicating the module on each device. 
            # Each replica handles a portion of the input, and then the gradients from each replica are summed into the original module.
            self.model = torch.nn.DataParallel(module=self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        logger.info(f'Saving {self.config.checkpoint_path}')
        torch.save(raw_model.state_dict(), self.config.checkpoint_path) # Save the checkpoint

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, 'module') else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == 'train'
            # Put the model in training mode if is_train is True otherwise don't
            model.train(is_train)
            # Get either the training or the test dataset, and then use DataLoader to load the data
            data = self.train_dataset if is_train else self.test_dataset
            loader = torch.utils.data.dataloader.DataLoader(
                dataset=data, shuffle=True, pin_memory=True, batch_size=config.batch_size, num_workers=config.num_workers
            )
            
            losses = []
            progress_bar = tqdm(iterable=enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

            # Get the index of iteration and the data
            for it, (x, y) in progress_bar:
                # Place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward the model
                with torch.set_grad_enabled(is_train):  # Set gradients if we are training otheerwise deactivate them
                    logits, loss = model(x, y)  # Get the output ("logits") and the loss from the model
                    loss = loss.mean()          # Collapse all losses if they are scattered on multiple GPUs
                    losses.append(loss.item())

                if is_train:
                    # Backpropagation and parameters update
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip) # Apply gradient clipping -> a technique used to prevent exploding gradients
                    optimizer.step()

                    # Decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()  # Number of tokens processed this step (i.e. label is not -100)
                        # Check if we are still in the warmup phase
                        if self.tokens < config.warmup_tokens:
                            # Linear warmup -> define amount of learning rate increase
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # Cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        # Apply modification to learning rate (i.e. either increase or decay)
                        lr = config.learning_rate * lr_mult

                        # Update the value of the learning rate in the optimizer
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # Report progress
                    progress_bar.set_description(f'Epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr{lr:e}')

            # If we are testing -> calculate and return loss
            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info('Test loss %f', test_loss)
                return test_loss
            
        best_loss = float('-inf')
        self.tokens = 0  # Counter used for learning rate decay

        for epoch in range(config.max_epochs):
            # Train
            run_epoch('train')
            # Test (if test dataset available)
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
            
            # Supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if good_model and self.config.checkpoint_path is not None:
                best_loss = test_loss
                self.save_checkpoint()
