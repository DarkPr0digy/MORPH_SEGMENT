'''
    Class for input and output languages, in this case it would be the source and target (segmented) words
'''

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
