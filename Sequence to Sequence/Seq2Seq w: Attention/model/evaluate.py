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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

MAX_LENGTH = 40
SOS_token = 0
EOS_token = 1
PAD_token = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class evaluate:

    def __init__(self, input_lang, output_lang):
        self.input_lang = input_lang
        self.output_lang = output_lang

    def showPlot(self, points):
        plt.plot(points)
        plt.show()

    def removeTags(self, segments):
        ortho = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", segments)
        ortho = ortho.replace("[]", "-")[:-1]

        return ortho

    def indexesFromWord(self, lang, word):
        return [lang.char2index[char] for char in list(word)]

    def tensorFromWord(self, lang, word):
        indexes = self.indexesFromWord(lang, word)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def tensorsFromPair(self, pair):
        input_tensor = tensorFromWord(input_lang, pair[0])
        target_tensor = tensorFromWord(output_lang, pair[1])
        return (input_tensor, target_tensor)
    '''
        Evaluates model between epochs
    '''
    def evaluateModel(self, encoder, decoder, sentence, max_length=MAX_LENGTH):
        with torch.no_grad():
            input_tensor = self.tensorFromWord(self.input_lang, sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.initHidden()

            encoder_outputs = torch.zeros(
                max_length, encoder.hidden_size, device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)

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
                    decoded_words.append(
                        self.output_lang.index2char[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]
    '''
        Evaluates words in the validation set randomly
    '''
    def evaluateRandomly(self, encoder, decoder, pairs, n=100,):
        for i in range(n):
            pair = random.choice(pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words = self.evaluateModel(encoder, decoder, pair[0])

            output = ''
            for char in output_words[0]:
                output += char

            print('<', output[:-1])
            print('')

    '''
        Evaluates words from the test set 
    '''
    def evalWords(self, encoder, decoder, n, printWords=False):
        lines = open('../../Data/Zulu/zulu.clean.test.conll', encoding='utf-8').\
            read().strip().split('\n')
        data = []
        correct = 0
        for line in lines:
            line = line.split(" | ")
            ortho = self.removeTags(line[2])
            data.append(line[0] + "\t" + ortho)

        for i in range(n):
            source, target = data[i].split('\t')
            output_words = self.evaluateModel(encoder, decoder, source.lower())
            output = ''
            for char in output_words[0]:
                output += char

            output = output[:-1]
            if target == output:
                correct += 1
            if printWords == True:
                print('>', source)
                print('=', target)
                print('<', output)
                print('')

        print("Accuracy: ", correct/n)
        return (correct/n)
    
    def showAttention(self, input_sentence, output_words, attentions):
        
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

    '''
        Shows attention for a segmented word
    '''
    def evaluateAndShowAttention(self, input_sentence, encoder, attn_decoder):
        output_words, attentions = self.evaluateModel(
            encoder, attn_decoder, input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        self.showAttention(input_sentence, output_words, attentions)





