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
import re
import Lang


class data:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SOS_token = 0
    EOS_token = 1
    PAD_token = 2
    MAX_LENGTH = 40
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

   

    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        return s

    def printWord(self):
        print("Word")
    '''
        Reads the datasets and executes preprocessing
    '''
    def readLangs(self, lang1, lang2, reverse=False):
        print("Reading lines...")

        lines = open('../../Data/Zulu/zulu.clean.train.conll', encoding='utf-8').\
            read().strip().split('\n')
        data = []
        for line in lines:
            line = line.split(" | ")
            ortho = self.removeTags(line[2])
            data.append(line[0] + "\t" + ortho)

        pairs = [[self.normalizeString(s)
                  for s in l.split('\t')] for l in data]

        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang.Lang(lang2)
            output_lang = Lang.Lang(lang1)
        else:
            input_lang = Lang.Lang(lang1)
            output_lang = Lang.Lang(lang2)

        return input_lang, output_lang, pairs


    def filterPair(self, p):
        return p

    def filterPairs(self, pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    def removeTags(self, segments):
        ortho = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", segments)
        ortho = ortho.replace("[]", "-")[:-1]

        return ortho
    '''
        Loads data from the dataset, intialises both source and target languages
    '''
    def prepareData(self, lang1, lang2, reverse=False):
        input_lang, output_lang, pairs = self.readLangs(lang1, lang2, reverse)
        print("Read %s sentence pairs" % len(pairs))
        pairs = self.filterPairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            input_lang.addWord(pair[0])
            output_lang.addWord(pair[1])
        print("Counted words:")
        print(input_lang.name, input_lang.n_chars)
        print(output_lang.name, output_lang.n_chars)
        return input_lang, output_lang, pairs


d = data()
input_lang, output_lang, pairs = d.prepareData('zulu', 'segmented', False)
print(input_lang.index2char)
print(output_lang.char2index)

print(random.choice(pairs))
