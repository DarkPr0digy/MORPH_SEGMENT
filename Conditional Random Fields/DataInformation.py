import os
import sys

import Levenshtein as LevenshteinDistance


class DataInformation:
    """
    This is a class that allows me to get information and numbers regarding the data after the
    pre-processing has been performed
    """
    def __init__(self, language: str):
        self.language = language
        self.input_files = ["morphology/" + language + '/' + language + ".clean.train.conll",
                            "morphology/" + language + '/' + language + ".clean.test.conll"]
        self.all_files = ["morphology/" + language + '/' + language + ".clean.train.conll",
                          "morphology/" + language + '/' + language + ".clean.dev.conll",
                          "morphology/" + language + '/' + language + ".clean.test.conll"]
        self.test_labels = []
        self.test_segments = []
        self.train_labels = []
        self.train_segments = []
        self.getSegmentsAndLabels()

    def getSegmentsAndLabels(self):
        """"
        Populates the lists that belong to the instance so that they can be used for further operations
        """
        counter = 0
        for file in self.input_files:
            input_file = open(os.path.join(sys.path[0], file), 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")
                surface_form = content[1].split('-')
                label_list = get_ortho_labels(content[2])

                if counter == 0:
                    for segment in surface_form:
                        if segment not in self.train_segments:
                            self.train_segments.append(segment)

                    for label in label_list:
                        if label not in self.train_labels:
                            self.train_labels.append(label)
                else:
                    for segment in surface_form:
                        if segment not in self.test_segments:
                            self.test_segments.append(segment)

                    for label in label_list:
                        if label not in self.test_labels:
                            self.test_labels.append(label)
            counter += 1

    def getSegmentsDictionary(self):
        """
        Method that cretes dictionaries for all words and their associated labesl
        :return: dictionary with key being surface form and entry being the associated labels
        """
        train_dict = {}
        test_dict = {}
        counter = 0
        for file in self.input_files:
            input_file = open(os.path.join(sys.path[0], file), 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")
                surface_form = content[1].split('-')
                label_list = get_ortho_labels(content[2])

                if counter == 0:
                    for i in range(len(surface_form)):
                        if surface_form[i] in train_dict:
                            if label_list[i] not in train_dict[surface_form[i]]:
                                train_dict[surface_form[i]].append(label_list[i])
                        else:
                            train_dict[surface_form[i]] = [label_list[i]]
                else:
                    for i in range(len(surface_form)):
                        if surface_form[i] in test_dict:
                            if label_list[i] not in test_dict[surface_form[i]]:
                                test_dict[surface_form[i]].append(label_list[i])
                        else:
                            test_dict[surface_form[i]] = [label_list[i]]
            counter += 1
        return train_dict, test_dict

    def missing_test_segments(self):
        """
        Method that takes out all segments in the test set that are not present in the training set
        :return: list of all segments in test set not in training set
        """
        missing_segments = []
        # print(sorted(self.train_segments))
        # print(sorted(self.test_segments))
        for segment in self.test_segments:
            if segment not in self.train_segments:
                missing_segments.append(segment)
        return missing_segments

    def missing_test_labels(self):
        """
        Method that takes out all labels in the test set that are not present in the training set
        :return: list of all labels in test set not in training set
        """
        missing_labels = []
        # print(sorted(self.train_labels))
        # print(sorted(self.test_labels))
        for label in self.test_labels:
            if label not in self.train_labels:
                missing_labels.append(label)
        return missing_labels

    def labels_and_segments(self):
        """
        finds strings that are present as both labels and segments in a training and test set.
        :return: list of strings that are both labels and segments
        """
        both = []
        for label in self.train_labels:
            if label in self.train_segments:
                both.append(label)
        for label in self.test_labels:
            if label in self.test_segments:
                both.append(label)
        return both

    def segments_ortho(self):
        """
        Gives Numbers to determine how functional method to generate surface segmentation form is
        :return: number of total segments that had to be edited and number of those segments that matches
        segments in the orthographic form
        """
        segments, match = 0, 0
        for file in self.all_files:
            input_file = open(os.path.join(sys.path[0], file), 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")
                surface = content[1].split("-")
                ortho = removeLabels(content[3]).split("-")
                orthographic = de_segment(removeLabels(content[3]))
                word = content[0]
                if not word.__eq__(orthographic):
                    for segment in surface:
                        if segment in ortho:
                            match += 1
                    segments += len(ortho)
        return segments, match

    def operations(self):
        """
        Metrics to determine the functionality of the surface segmentation generator
        :return: number of edited words, total words in dataset, number of operations performed, number of deletions,
        number of replacements
        """
        edited_words, total_words, operations, delete, replace = 0, 0, 0, 0, 0
        for file in self.all_files:
            input_file = open(os.path.join(sys.path[0], file), 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")
                orthographic = de_segment(removeLabels(content[3]))
                word = content[0]
                total_words += 1
                if not word.__eq__(orthographic):
                    edited_words += 1
                    edits = LevenshteinDistance.editops(orthographic, word)
                    for ed in edits:
                        if ed[0] == 'delete':
                            operations += 1
                            delete += 1
                        elif ed[0] == 'replace':
                            operations += 1
                            replace += 1
        return edited_words, total_words, operations, delete, replace


def get_ortho_labels(orthographic: str):
    """
    Method to get the labels from the orthographic form of the word
    :param orthographic: the orthographic form of the word with labels included
    :return: a list of the labels in the word
    """
    labels = []
    tmp = ''
    tag = False

    # Get all labels from orthographic form
    for char in orthographic:
        if char == '[':
            tag = True
        elif char == ']':
            labels.append(tmp)
            tag = False
            tmp = ''
        elif tag:
            tmp += char
    return labels


def get_surface_segments(orthographic: str):
    """
    Method to extract the segments from the orthographic form of the word
    :param orthographic: the orthographic form of the word
    :return: list of all the segments in the word
    """
    segments = []
    tmp = ''
    label = False

    # Get all segments from orthographic form
    for char in orthographic:
        if char == '[':
            segments.append(tmp)
            tmp = ''
            label = True
        elif char == ']':
            label = False
        elif not label:
            tmp += char
    return segments


def de_segment(word: str):
    """
    Method used to de-segment the orthographic form of a word
    :param word: orthographic form of the word without labels
    :return: orthographic word as one string without segments
    """
    ans = ""
    for char in word:
        if char != "-":
            ans += char
    return ans


def removeLabels(str2: str):
    """
    Method to remove labels from the orthographic segmentation so this form
    can be used to generate the surface segmentation
    :param str2: orthographic form
    :return: segmented orthographic form of word
    """
    str2_arr = []
    last_seen_bracket = []
    for char in str2:
        if char == "[":
            last_seen_bracket.append(char)
            str2_arr.append("-")
        elif char == "(":
            last_seen_bracket.append(char)
        elif char == ")" or char == "]":
            if len(last_seen_bracket) >= 1:
                last_seen_bracket.pop()
            else:
                continue
        elif char == "-" or char == '$':
            continue
        elif len(last_seen_bracket) >= 1:
            continue
        else:
            str2_arr.append(char)

    if len(str2_arr) > 1:
        for i in range(len(str2_arr)):
            try:
                if str2_arr[i] == "-" and str2_arr[i - 1] == "-":
                    str2_arr.pop(i - 1)
                    # Some segments have dual purpose, so this removes dual dashes that result from this
            except IndexError:
                continue

        if str2_arr[len(str2_arr) - 1] == "\n":
            str2_arr.pop()

    return "".join(str2_arr).rstrip("-").lstrip("-")
