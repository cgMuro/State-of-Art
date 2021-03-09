# https://github.com/karpathy/minGPT

"""
Model Architecture:
    1. Combination of token encoding and positional encoding
    2. Uniform sequence of Transformer blocks:
        * each transformer = 1-hidden-layer MLP block + self-attention block
        * all blocks feed into a central residual pathway
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


# ----------------------------------- BUILD MODEL ----------------------------------- #

class GPTConfig:
    """ Base GPT configuration. Params are common to all GPT versions """
    embeddings_dropout = 0.1
    residual_dropout = 0.1
    attention_dropout = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size

        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network (roughly 125M params) """
    n_layers = 12
    n_heads = 12
    n_embeddings = 768


class CausalSelfAttention(nn.Module):
    """ A vanilla multi-head masked self-attention layer with a projection at the end """

    def __init__(self, config):
        super().__init__()
        assert config.n_embeddings % config.n_heads == 0
        # Key, query, value projections for all heads
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
        B, T, C = x.size()

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dimension
        k = self.key(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)   # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        # Casual self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attention = attention.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)
        y = attention @ v   # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side

        # Output projection
        y = self.residual_dropout(self.projection(y))
        return y


class Block(nn.Module):
    """ An unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embeddings)
        self.ln2 = nn.LayerNorm(config.n_embeddings)
        self.attention = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embeddings, 4 * config.n_embeddings),
            nn.GELU(),
            nn.Linear(4 * config.n_embeddings, config.n_embeddings),
            nn.Dropout(config.residual_dropout)
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """ The full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # Input embedding stem
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embeddings)
        self.position_embeddings = nn.Parameter(torch.zeros(1, config.block_size, config.n_embeddings))
        self.dropout = nn.Dropout(config.embeddings_dropout)
        # Transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        # Decoder head
        self.ln_f = nn.LayerNorm(config.n_embeddings)
        self.head = nn.Linear(config.n_embeddings, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # Separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # Full param name
                
                if pn.endswith('bias'):
                    # All biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # Weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # Weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # Special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('position_embeddings')

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
        b, t = idx.size()
        assert t <= self.block_size, 'Cannot forward, model block size is exhausted'

        # Forward the GPT model
        token_embeddings = self.token_embeddings(idx)             # Each index maps to a (learnable) vector
        position_embeddings = self.position_embeddings[:, :t, :]  # Each position maps to a (learnable) vector
        x = self.dropout(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # If wee are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


# ----------------------------------- TRAINER ----------------------------------- #

class TrainerConfig:
    # Optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # Only  applied on matrix multiplication weights
    # Learning rate decay params -> linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9 # At what point we reach 10% of original learning rate
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
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw  model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        logger.info(f'Saving {self.config.checkpoint_path}')
        torch.save(raw_model.state_dict(), self.config.checkpoint_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, 'module')  else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = torch.utils.data.dataloader.DataLoader(
                data, shuffle=True, pin_memory=True, batch_size=config.batch_size, num_workers=config.num_workers
            )
            
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

            for it, (x, y) in pbar:
                # Place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = loss.mean()  # Collapse all losses if they are scattered on multiple GPUs
                    losses.append(loss.item())

                if  is_train:
                    # Backpropagation and parameters update
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # Decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()  # Number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # Linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # Cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult

                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # Report progress
                    pbar.set_description(f'Epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr{lr:e}')

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info('Test loss %f', test_loss)
                return test_loss
            
        best_loss = float('-inf')
        self.tokens = 0  # Counter used for learning rate decay

        for epoch in range(config.max_epochs):
            run_epoch('train')

            if self.test_dataset is not None:
                test_loss = run_epoch('test')
            
            # Supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if good_model and self.config.checkpoint_path is not None:
                best_loss = test_loss
                self.save_checkpoint()
