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
class evaluate:

    def __init__(self, model, ):
        self.model = model

    '''
        Function to evaluate the model inbetween each epoch
    '''
    def evaluateModel(self, model, iterator, criterion):
    
        model.eval()
        
        epoch_loss = 0
        
        with torch.no_grad():
        
            for i, batch in enumerate(iterator):
                
                src = batch.src
                trg = batch.trg

                output, _ = model(src, trg[:,:-1])
                
                #output = [batch size, trg len - 1, output dim]
                #trg = [batch size, trg len]
                
                output_dim = output.shape[-1]
                
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:,1:].contiguous().view(-1)
                
                #output = [batch size * trg len - 1, output dim]
                #trg = [batch size * trg len - 1]
                
                loss = criterion(output, trg)

                epoch_loss += loss.item()
            
        return epoch_loss / len(iterator)

    def translate_sentence(self, sentence, src_field, trg_field, model, device, max_len = 50):
        
        model.eval()
            
        if isinstance(sentence, str):
            nlp = spacy.load('de')
            tokens = [token.text.lower() for token in nlp(sentence)]
        else:
            tokens = [token.lower() for token in sentence]

        tokens = [src_field.init_token] + tokens + [src_field.eos_token]
            
        src_indexes = [src_field.vocab.stoi[token] for token in tokens]

        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
        
        src_mask = model.make_src_mask(src_tensor)
        
        with torch.no_grad():
            enc_src = model.encoder(src_tensor, src_mask)

        trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

        for i in range(max_len):

            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

            trg_mask = model.make_trg_mask(trg_tensor)
            
            with torch.no_grad():
                output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            
            pred_token = output.argmax(2)[:,-1].item()
            
            trg_indexes.append(pred_token)

            if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
                break
        
        trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
        
        return trg_tokens[1:], attention

    '''
        Function to display attention over the segmented word
    '''
    def display_attention(self, sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2):
        
        assert n_rows * n_cols == n_heads
        
        fig = plt.figure(figsize=(15,25))
        
        for i in range(n_heads):
            
            ax = fig.add_subplot(n_rows, n_cols, i+1)
            
            _attention = attention.squeeze(0)[i].cpu().detach().numpy()

            cax = ax.matshow(_attention)

            ax.tick_params(labelsize=12)
            ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                            rotation=45)
            ax.set_yticklabels(['']+translation)

            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()
        plt.close()

    


    def arrToWord(self, translation):
        out = ""
        for letter in translation:
            out += letter
        return out
    '''
        Function to evaluate words in the test set
    '''
    def evaluateWords(self, n, test_data, SRC, TRG, model, device, printWords=True, save=False):
        file = open('results.txt', 'w')
        for i in range(n):
            #example_idx = random.randint(0, len(test_iterator))
            src = vars(test_data.examples[i])['src']
            trg = vars(test_data.examples[i])['trg']
            
            translation, attention = self.translate_sentence(src, SRC, TRG, model, device)
            if printWords==True:
                print("Target:\t", self.arrToWord(trg))
                print("Prediction:\t", self.arrToWord(translation))
                print("")
            if save==True:
                file.write(self.arrToWord(trg)+'\t'+self.arrToWord(translation).replace("<eos>", "")+'\n')
        file.close()

