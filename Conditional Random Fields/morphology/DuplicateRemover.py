import os
import sys

from morphology.Symbols import Symbols


class DuplicateRemover:
    """
    Class that will handle the removal of words in training set and test set from training set
    """

    def __init__(self, language: str):
        """
        Constructor for class
        :param language: string of the language
        """
        test = language + '/' + language + '.test.conll'
        train = language + '/' + language + '.train.conll'
        out = language + '/' + language + '.unique.train.conll'
        self.test_file = open(os.path.join(sys.path[0], test), "r")
        self.training_file = open(os.path.join(sys.path[0], train), "r")
        self.output_file = open(os.path.join(sys.path[0], out), "w")
        self.s = Symbols()

    def removeDuplicates(self):
        """Method that will remove all items that occur in both test and training sets from the training sets to
        avoid over fitting"""
        testWords = []
        firstLine = True
        for line in self.test_file.readlines():
            if firstLine:
                firstLine = False
                continue

            tmp = line.split("\t")[0]
            if not self.s.inArr(tmp):
                testWords.append(tmp)

        self.test_file.close()

        trainingWords = []
        firstLine = True
        for line in self.training_file.readlines():
            if firstLine:
                firstLine = False
                continue

            trainingWords.append(line)

        self.training_file.close()

        for testWord in testWords:
            for i in range(len(trainingWords)):
                try:
                    if testWord.lower() == trainingWords[i].split("\t")[0].lower():
                        del trainingWords[i]
                except IndexError:
                    continue

        for line in trainingWords:
            self.output_file.write(line)
        self.output_file.close()
