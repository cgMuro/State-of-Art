# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# Data: https://download.pytorch.org/tutorial/data.zip

# In sequence to sequence neural networks, we have 2 networks, one encoder and one decoder.
# Encoder network -> condenses an input sequence into a vector
# Decoder network -> unfolds the vector (from the encoder network) into a new sequence
# The attention mechanism is a technique that let's the decoder network focus on a specific range of input sequence


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# If available set device to cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------------------------- PREPARE DATA ----------------------------------- #

SOS_TOKEN = 0
EOS_TOKEN = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}   # Maps words to indexes
        self.word2count = {}   # Maps each word to the number of times it appears
        self.index2word = {0: 'SOS', 1: 'EOS'}  # Maps indexes to words
        self.n_words = 2   # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Since the files are all in Unicode we will translate them to ASCII
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# Make everything lowercase and trim the puctuation
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]', r' ', s)
    return s


# Each line in the file is composed by the english phrase and the translation separated by tabs
# ex: I am cold.    J'ai froid.
# So we want to loop over each line and normalize each string in the line. 
# Then we use the "reverse" parameter to decide from what to what language we want the translation to be.
def read_langs(lang1, lang2, reverse=False):
    print('Reading lines...')

    # Read file and split into lines
    lines = open(f'data/{lang1}-{lang2}.txt', encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for pair in line.split('\t')] for line in lines]

    # Reverse pairs
    if reverse:
        # We switch the two phrases to change the translation from what to what language
        pairs = [list(reversed(p)) for p in pairs]
        # We create a Lang object for each of the languages
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        # We create a Lang object for each of the languages
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs



# Optional filtering
# If we want to train quickly, we can trim the pairs to sentences of less than 10 words and that starts with some prefixes
MAX_LENGTH = 10
ENGLISH_PREFIXIES = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filter_pair(pair):
    return len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH and pair[1].startswith(eng_prefixes)
def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

# The full prepare data function:
    # read text file and split into lines, split lines into pairs
    # normalize text, filter by length and content 
    # make word lists from sentences in pairs
def prepare_data(lang1, lang2, reverse=False, filter=False):
    # Read data and return 2 Langs objects and the normalized pairs
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print(f'Read {len(pairs)} sentence pairs')

    if filter:
        # Filter the pairs to reduce data
        pairs = filter_pairs(pairs)
        print(f'Trimmed to {len(pairs)} sentence pairs')

    # Count the words and update the attributes on Lang objects
    print('Counting words...')
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print('Counted words:')
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs


# Get data
input_lang, output_lang, pairs = prepare_data('eng', 'fra', True, False)
print(random.choice(pairs))



# ----------------------------------- MODEL ----------------------------------- #

# A Sequence to Sequence Network (seq2seq) or Encoder Decoder network is a model composed by 2 RNNs, an encoder and a decoder.
# The encoder reads the input sequence and outputs a vector. The decoder reads the vector and outputs a sequence.
# The advantage of this system with respect to a single RNNs is that in this way the output sequence length and order are not dictated by the input sequence.

# THE ENCODER
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# THE DECODER
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# ATTENTION ENCODER

# Attention allows the decoder network to “focus” on a different part of the encoder’s outputs for every step of the decoder’s own outputs.
# Process:
#   1. calculate a set of attention weights
#   2. multiply by the encoder output vectors to create a weighted combination
#   3. the result contains information about that specific part of the input sequence (helps the decoder choose the right output words)

#Calculating the attention weights is done with another feed-forward layer attn, using the decoder’s input and hidden state as inputs. 
#Because there are sentences of all sizes in the training data, to actually create and train this layer we have to choose a maximum sentence length (input length, for encoder outputs) that it can apply to. 
#Sentences of the maximum length will use all the attention weights, while shorter sentences will only use the first few.
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)),
            dim=1
        )

        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# ----------------------------------- TRAINING ----------------------------------- #

# Prepare data

# Get input tensor (indexes of the words in the input sentence)
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

# Get target tensor (indexes of the words in the target sentence)
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_TOKEN) # Append the EOS token
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# Training the model
# 1. Run the input sentence through the encoder
# 2. Keep track of every output and the latest hidden state
# 3. The decoder is given the <SOS> token as its first input, and the last hidden state of the encoder as its first hidden state


teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    
    decoder_input = torch.tensor([[SOS_TOKEN]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]   # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach() # Detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_TOKEN:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length



# Helper function to print time elapsed and estimated time remaining given the current time and progress %.
import time
import math

def asMinutes(s):
    m = math.floor(s/60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s  = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(res))


# Training process
# 1. Start timer
# 2. Initialize optimizers and criterion
# 3. Create set of training pairs
# 4. Start empty losses array for plotting

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorFromPair(random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

# Plotting results
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # This locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# ----------------------------------- EVALUATION ----------------------------------- #

# We feed the decoder’s predictions back to itself for each step and every time the it predicts a word we add it to the output string, and if it predicts the EOS token we stop there.
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

# We can evaluate random sentences from the training set and print out the input, target, and output to make some subjective quality judgements:
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# ----------------------------------- Training and Evaluating ----------------------------------- #

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout=0.1).to(device)

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

evaluateRandomly(encoder1, attn_decoder1)

# Visualizing Attention

# The attention mechanism has highly interpretable outputs. Because it is used to weight specific encoder outputs of the input sequence, we can imagine looking where the network is focused most at each time step.
def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)
    
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluate_and_show_attention(input_sentence):
    output_words, attentions = evaluate(encoder1, attn_decoder1, input_sentence)
    print('Input =', input_sentence)
    print('Output = ', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)


evaluate_and_show_attention("elle a cinq ans de moins que moi .")
evaluate_and_show_attention("elle est trop petit .")
evaluate_and_show_attention("je ne crains pas de mourir .")
evaluate_and_show_attention("c est un jeune directeur plein de talent .")
