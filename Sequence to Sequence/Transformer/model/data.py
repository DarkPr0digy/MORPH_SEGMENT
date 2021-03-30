import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time

SEED = 1234
class data:
    BATCH_SIZE = 32
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def tokenize(text):
        """
        Tokenizes all words into characters, this achieves character-level tokenization
        """
        return [char for char in text]

    SRC = Field(tokenize = tokenize, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True, 
                batch_first = True)

    TRG = Field(tokenize = tokenize, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True, 
                batch_first = True)
    '''
        Returns iterators for model to run on, data sets and fields
    '''
    def getIterators(self):
        fields = {'src': ('src', self.SRC), 'trg': ('trg', self.TRG)}
        train_data, valid_data, test_data = torchtext.data.TabularDataset.splits(
                                path = "../../Data/zulu/json",
                                train = 'zulu-train.json',
                                test = 'zulu-test.json',
                                validation = 'zulu-validation.json',
                                format = 'json',
                                fields = fields)
        self.SRC.build_vocab(train_data)
        self.TRG.build_vocab(train_data)

        train_iterator, valid_iterator, test_iterator = torchtext.data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        sort_key = lambda x: x.src, 
        batch_size=self.BATCH_SIZE,
        device=self.device)

        print("Train Size: ", len(train_data))
        print("Test Size: ", len(test_data))
        print("Valid: ", len(valid_data))
        print("Source Vocab: ", self.SRC.vocab.itos)
        print("Target Vocab: ", self.TRG.vocab.itos)

        return train_iterator, valid_iterator, test_iterator, test_data, train_data, self.SRC, self.TRG
    


