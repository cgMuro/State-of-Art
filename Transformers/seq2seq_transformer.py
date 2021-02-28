# https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import math
import time
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator



# ----------------------------------- BUILD MODEL ----------------------------------- #

# The language modeling task is to assign a probability for the likelihood of a given word (or a sequence of words) to follow a sequence of words. 
# A sequence of tokens are passed to the embedding layer first, followed by a positional encoding layer to account for the order of the word. 
# The "TransformerEncoder" consists of multiple layers of "TransformerEncoderLayer". Along with the input sequence, a square attention mask 
# is required because the self-attention layers in "TransformerEncoder" are only allowed to attend the earlier positions in the sequence. 
# Any tokens on the future positions should be masked. To have the actual words, the output of "TransformerEncoder" model is sent to 
# the final Linear layer, which is followed by a log-Softmax function.

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, flaot(0.0))
        return mask

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output



# PositionalEncoding module injects information about the relative or absolute position of the tokens in the sequence. 
# The positional encodings have the same dimension as the embeddings so that the two can be summed.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len,  d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,  d_model, 2).float() *  (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

 

# ----------------------------------- LOAD AND BATCH DATA ----------------------------------- #


url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
test_filepath, valid_filepath, train_filepath = extract_archive(download_from_url(url))
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, iter(io.open(train_filepath, encoding='utf8'))))

def data_process(raw_text_iter):
    data = [torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_data = data_process(iter(io.open(train_filepath, encoding='utf8')))
val_data = data_process(iter(io.open(valid_filepath, encoding='utf8')))
test_data = data_process(iter(io.open(test_filepath, encoding='utf8')))

# Get device -> train on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# batchify -> arranges the dataset into columns, trimming off any tokens remaining after the data has been divided into batches
def batchify(data, bsz):
    # Divide the dataset into bsz parts
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders)
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)


# Generate input and target sequence #
# Init chucks' length
bptt = 35
# get_batch -> generates the input and target sequence for the transformer model.
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


# Initiate and instance #

# Model's parameters set up
n_tokens = len(vocab.stoi)   # Size of vocabulary
embedding_size = 200         # Embedding dimension
n_hid = 200                  # Dimension of the feedforward network model in nn.TransformerEncoder
n_layers = 2                 # Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
n_head = 2                   # Number of heads in the multiheadattention models
dropout = 0.2                # Dropout value

model = TransformerModel(ntoken=n_tokens, ninp=embedding_size, nhead=n_head, nhid=n_hid, nlayers=n_layers, dropout=dropout).to(device)



# ----------------------------------- TRAINING and EVALUATION ----------------------------------- #

# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer -> Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
# Learning Rate scheduler -> StepLR adjusts the learning rate through the epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train():
    # Turn the model on training mode
    model.train()
    total_loss = 0
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        if data.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, n_tokens), targets)
        loss.backward()
        # Scale all the gradients together to prevent exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss))
            )
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    # Turn the model on evaluation mode
    eval_model.eval()
    total_loss = 0
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            if data.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, src_mask)
            output_flat = output.view(-1, n_tokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


# TRAIN MODEL #
best_val_loss = float('inf')
epochs = 3
best_model = None

for epoch in  range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 90)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
    print('-' * 90)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()

# EVALUATE MODEL #
test_loss = evaluate(best_model, test_data)
print('=' * 90)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 90)
