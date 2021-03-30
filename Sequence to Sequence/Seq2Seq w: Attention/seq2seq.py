

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



SOS_token = 0
EOS_token = 1
PAD_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "<", 1: ">", 2: "+"}
        self.n_chars = 3  # Count SOS and EOS

    def addWord(self, word):
        for character in list(word):
            self.addCharacter(character)

    def addCharacter(self, character):
        if character not in self.char2index:
            self.char2index[character] = self.n_chars
            self.char2count[character] = 1
            self.index2char[self.n_chars] = character
            self.n_chars += 1
        else:
            self.char2count[character] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    #s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s



import re
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('../Data/Zulu/zulu.clean.train.conll' , encoding='utf-8').\
        read().strip().split('\n')
    data = []
    for line in lines:
        line = line.split(" | ")
        ortho = removeTags(line[2])
        data.append(line[0] + "\t" + ortho)

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in data]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs



MAX_LENGTH = 40

def filterPair(p):
    return p


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def removeTags(segments):
    ortho = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", segments)
    ortho = ortho.replace("[]","-")[:-1]

    return ortho


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addWord(pair[0])
        output_lang.addWord(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_chars)
    print(output_lang.name, output_lang.n_chars)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('zulu', 'train', False)
print(input_lang.index2char)
print(output_lang.char2index)

print(random.choice(pairs))



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_var = hidden_size // 2 if bidirectional else hidden_size
        self.n_layers = n_layers 
        self.n_directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional


        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size,self.hidden_var, num_layers=self.n_layers, bidirectional=self.bidirectional)

    def forward(self, input, hidden):
        #print("input", input.shape, "hidden", hidden.shape)
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        #print("output:: ", output.shape, "hidden:: ", hidden.shape)

        return output, hidden

    def initHidden(self):
        h_state = torch.zeros(self.n_layers * self.n_directions, 1, self.hidden_var)
        c_state = torch.zeros(self.n_layers * self.n_directions, 1, self.hidden_var)
        hidden = (h_state, c_state)
        return hidden




class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.3, max_length=MAX_LENGTH, n_layers=1, bidirectional= False):
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
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.n_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        #print("DECODER")
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        self.hidden = hidden 

        hidden_h_rows = ()
        hidden_c_rows = ()

        decoder_h, decoder_c = hidden
        hidden_shape = decoder_h.shape[0]

                # h_state
        for x in range(0, hidden_shape):
            hidden_h_rows += (decoder_h[x],)

                # c_state
        for x in range(0, hidden_shape):
            hidden_c_rows += (decoder_c[x],)

        if self.bidirectional:
            decoder_h_cat = torch.cat(hidden_h_rows, 1)
            # Make sure the h_dim size is compatible with num_layers with concatenation.
            decoder_h = decoder_h_cat.view((self.n_layers, 1, self.hidden_size))  # hidden_size=256   

            decoder_c_cat = torch.cat(hidden_c_rows, 1)
            decoder_c = decoder_c_cat.view((self.n_layers, 1, self.hidden_size))  # hidden_size=256
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



def indexesFromWord(lang, word):
    return [lang.char2index[char] for char in list(word)]


def tensorFromWord(lang, word):
    indexes = indexesFromWord(lang, word)
    # while(len(indexes)!=MAX_LENGTH-1):
    #     indexes.append(PAD_token)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromWord(input_lang, pair[0])
    target_tensor = tensorFromWord(output_lang, pair[1])
    return (input_tensor, target_tensor)

training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(1)]
#print(training_pairs[0])print(training_pairs[0][1].shape)
print(training_pairs[0][0].shape, training_pairs[0][1].shape)
for c in training_pairs[0][1]:
    print(output_lang.index2char[c.item()], end='')
    
print("")
for c in training_pairs[0][0]:
    print(input_lang.index2char[c.item()], end='')



teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    #print("InputTensor: ", input_tensor.shape)
    #print("TargeTensor: ", target_tensor.shape)
    for ei in range(input_length):
        #print(input_tensor[ei])
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
        #print(encoder_hidden.shape)

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            #print(decoder_output, target_tensor[di])
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            #print(topv, topi, target_tensor[di])
            loss += criterion(decoder_output, target_tensor[di])

            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() #/ target_length


import time
import math


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


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.001, batch_size=64):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.CrossEntropyLoss()#nn.NLLLoss()
    t = torch.zeros([MAX_LENGTH, 1], dtype=torch.long)
    for iter in range(1, n_iters + 1):
        

        training_pair = training_pairs[iter - 1]
        input_tensor  = training_pair[0]
        target_tensor = training_pair[1]
        #print(p.shape)
        #print(target_tensor.shape)

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


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np



import matplotlib.pyplot as plt

def showPlot(points):
    plt.plot(points)
    plt.show()



def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromWord(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('>')
                break
            else:
                decoded_words.append(output_lang.index2char[topi.item()])

            decoder_input = topi.squeeze().detach()
        #print(decoder_attentions[:di + 1].shape)
        return decoded_words, decoder_attentions[:di + 1]



def evaluateRandomly(encoder, decoder, n=100):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        #print(output_words[0], ' ', end='')
        output = ''
        for char in output_words[0]:
            output += char
        #output_sentence = ' '.join(output_words)
        #print(output_words[0][0])
        print('<', output[:-1])
        print('')

def evalWords(encoder, decoder, n, printWords=False):
    lines = open('../Data/Zulu/zulu.clean.test.conll', encoding='utf-8').\
        read().strip().split('\n')
    data = []
    correct = 0
    for line in lines:
        line = line.split(" | ")
        ortho = removeTags(line[2])
        data.append(line[0] + "\t" + ortho)
    # Split every line into pairs and normalize
    for i in range(n):
        source, target = data[i].split('\t')
        output_words = evaluate(encoder, decoder, source.lower())
        output = ''
        for char in output_words[0]:
            output += char
        #output_sentence = ' '.join(output_words)
        #print(output_words[0][0])
        output = output[:-1]
        if target==output:
          correct += 1
        if printWords==True:
            print('>', source)
            print('=', target)
            print('<', output)
            print('')

    print("Accuracy: ", correct/n)
    return (correct/n)

hidden_size = 256
iterations = len(pairs)
epochs = 20
encoder = EncoderRNN(input_lang.n_chars, hidden_size, 2, True).to(device)
attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_chars, dropout_p=0.3, n_layers=2, bidirectional=True).to(device)


print(encoder)
print(attn_decoder)
print(iterations)
print("Training Model...")
validationAccuracy = []
for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    loss = trainIters(encoder, attn_decoder, iterations, print_every=100)
    valAcc = evalWords(encoder, attn_decoder, 100, printWords=False)
    validationAccuracy.append(valAcc)

showPlot(loss)
showPlot(validationAccuracy)
evaluateRandomly(encoder, attn_decoder)

evalWords(encoder, attn_decoder, 100, printWords=True)



output_words, attentions = evaluate(
    encoder, attn_decoder, "elitholakala")
plt.matshow(attentions.numpy())



def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + list(input_sentence) +
                       ['>'])
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder, attn_decoder, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


evaluateAndShowAttention("elitholakala")

evaluateAndShowAttention("womphakathi")

evaluateAndShowAttention("nezimhlophe")

evaluateAndShowAttention("ngokobulili")

