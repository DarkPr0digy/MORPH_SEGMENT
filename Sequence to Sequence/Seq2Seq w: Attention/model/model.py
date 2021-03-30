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
import data
import evaluate
import time
import math

MAX_LENGTH = 40
SOS_token = 0
EOS_token = 1
PAD_token = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
    Encoder unit, specifies number of layers and bidirectionality
'''
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_var = hidden_size // 2 if bidirectional else hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, self.hidden_var,
                            num_layers=self.n_layers, bidirectional=self.bidirectional)

    def forward(self, input, hidden):

        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)

        return output, hidden

    def initHidden(self):
        h_state = torch.zeros(
            self.n_layers * self.n_directions, 1, self.hidden_var)
        c_state = torch.zeros(
            self.n_layers * self.n_directions, 1, self.hidden_var)
        hidden = (h_state, c_state)
        return hidden

'''
    Attention Decoder unit, specifies number of layers and bidirectionality, dropout 
'''
class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.3, max_length=MAX_LENGTH, n_layers=1, bidirectional=False):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(
            self.hidden_size, self.hidden_size, num_layers=self.n_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):

        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        self.hidden = hidden

        hidden_h_rows = ()
        hidden_c_rows = ()

        decoder_h, decoder_c = hidden
        hidden_shape = decoder_h.shape[0]

        for x in range(0, hidden_shape):
            hidden_h_rows += (decoder_h[x],)

        for x in range(0, hidden_shape):
            hidden_c_rows += (decoder_c[x],)

        if self.bidirectional:
            decoder_h_cat = torch.cat(hidden_h_rows, 1)

            decoder_h = decoder_h_cat.view(
                (self.n_layers, 1, self.hidden_size))

            decoder_c_cat = torch.cat(hidden_c_rows, 1)
            decoder_c = decoder_c_cat.view(
                (self.n_layers, 1, self.hidden_size))
            hidden_lstm = (decoder_h, decoder_c)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden_lstm[0][0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden_lstm)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        h_state = torch.zeros(self.n_layers * 1, 1, self.hidden_var)
        c_state = torch.zeros(self.n_layers * 1, 1, self.hidden_var)
        hidden = (h_state, c_state)
        return hidden


'''
    Functions to convert words into tensors
'''
def indexesFromWord(lang, word):
    return [lang.char2index[char] for char in list(word)]


def tensorFromWord(lang, word):
    indexes = indexesFromWord(lang, word)

    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromWord(input_lang, pair[0])
    target_tensor = tensorFromWord(output_lang, pair[1])
    return (input_tensor, target_tensor)


teacher_forcing_ratio = 0.5

'''
   Training function, specifies the source, target, encoder, deocoder + optimisers and the loss function
'''
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
       

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else:

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])

            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

'''
    Training iteration function, specifies the number of iterations
'''
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.001, batch_size=64):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.CrossEntropyLoss()  
    t = torch.zeros([MAX_LENGTH, 1], dtype=torch.long)
    for iter in range(1, n_iters + 1):

        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    return plot_losses

'''
    Specify the hidden size, number of iterations and epochs
'''
d = data.data()
input_lang, output_lang, pairs = d.prepareData('zulu', 'segmented', False)
hidden_size = 256
iterations = len(pairs)
epochs = 20
encoder = EncoderRNN(input_lang.n_chars, hidden_size, 2, True).to(device)
attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_chars,
                              dropout_p=0.3, n_layers=2, bidirectional=True).to(device)


print(encoder)
print(attn_decoder)
print(iterations)
print("Training Model...")
validationAccuracy = []

'''
    Training loop over N_Epochs
'''
for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    loss = trainIters(encoder, attn_decoder, iterations, print_every=100)
    e = evaluate.evaluate(input_lang, output_lang)
    valAcc = e.evalWords(encoder, attn_decoder, 100, printWords=False)
    validationAccuracy.append(valAcc)
    e.showPlot(loss)
    e.showPlot(validationAccuracy)
    e.evaluateRandomly(encoder, attn_decoder, pairs)
    e.evalWords(encoder, attn_decoder, 100, printWords=True)
    e.evaluateAndShowAttention("elitholakala", encoder, attn_decoder)
