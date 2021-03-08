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
import torch
import torch.nn as nn
from torch.nn import functional as F

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
