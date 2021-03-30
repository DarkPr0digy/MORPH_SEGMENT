import os
import sys
import time
import pickle

import sklearn_crfsuite
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer


class BaselineCRF:
    """
    Baseline CRF implemented using sklearncrf_suite
    """

    def __init__(self, language: str):
        """
        The constructor function for the baseline CRF that takes in the name of the language and looks for the file
         corresponding to that language in the morphology folder
        :param language: string of the language that particular model should focus on
        """
        self.input_files = ["../morphology/" + language + '/' + language + ".clean.train.conll",
                            "../morphology/" + language + '/' + language + ".clean.dev.conll",
                            "../morphology/" + language + '/' + language + ".clean.test.conll"]
        self.language = language

    def surface_segmentation(self):
        """
        This method makes use of sklearn crfsuite to perform surface segmentation
        :return: list of predicted segments and list of correct segments
        """
        tic = time.perf_counter()
        # Collect the Data
        ##################################################
        training_data, dev_data, test_data = {}, {}, {}
        dictionaries = (training_data, dev_data, test_data)
        counter = 0
        for file in self.input_files:
            input_file = open(os.path.join(sys.path[0], file), 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")
                result = []
                morph = ''
                tag = False

                for char in content[1]:
                    # Surface Segmentation
                    if char == '-':
                        result.append(morph)
                        morph = ''
                    else:
                        morph += char

                if morph.strip():
                    result.append(morph.strip())

                label = ''
                for morph in result:
                    if len(morph) == 1:
                        label += 'S'
                    else:
                        label += 'B'
                        for i in range(len(morph) - 2):
                            label += 'M'
                        label += 'E'

                # current dictionary being referenced
                # Key is word and value is segmented form
                # print(content)
                dictionaries[counter][content[0]] = label

            # input_file.close()
            counter += 1
        toc = time.perf_counter()
        print("Data Collected in " + str(tic - toc.__round__(2)))

        # Compute Features & Optimise Model Using Dev Set
        ##################################################
        best_epsilon, best_max_iteration = 0, 0
        maxF1 = 0
        print("Beginning Feature Computation and Model Optimisation")
        tic = time.perf_counter()

        '''for epsilon in [0.001, 0.00001, 0.0000001]:
            for max_iterations in [80, 120, 160]:
                X_training, Y_training, words_training = surface_segment_data_preparation(training_data)
                X_dev, Y_dev, words_dev = surface_segment_data_preparation(dev_data)
                crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=epsilon, max_iterations=max_iterations)
                crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)

                Y_predict = crf.predict(X_dev)
                Y_dev = MultiLabelBinarizer().fit_transform(Y_dev)
                Y_predict = MultiLabelBinarizer().fit_transform(Y_predict)
                f1 = f1_score(Y_dev, Y_predict, average='micro')
                if f1 > maxF1:
                    f1 = maxF1
                    best_epsilon = epsilon
                    best_max_iteration = max_iterations

        print(best_max_iteration)
        print(best_epsilon)'''

        toc = time.perf_counter()
        print("Features Successfully Computed & Model Optimised " + str(tic - toc.__round__(2)))

        # Evaluate Model On the Test Set Using Optimised Model
        #######################################################

        best_max_iteration = 160
        best_epsilon = 1e-07

        # a, b, c = surface_segment_data_preparation(training_data)
        # print("X_Training: " + str(a[len(a) - 1]) + "\n################################")
        # print("Y_training: " + str(b[len(b) - 1]) + "\n################################")
        # print("Words Training: " + str(c[len(c) - 1]) + "\n############################")

        X_training, Y_training, words_training = surface_segment_data_preparation(training_data)
        X_dev, Y_dev, words_dev = surface_segment_data_preparation(dev_data)
        X_test, Y_test, words_test = surface_segment_data_preparation(test_data)
        crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=best_epsilon, max_iterations=best_max_iteration)
        crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)

        Y_predict = crf.predict(X_test)

        return Y_predict, Y_test

    def to_use_surface_crf(self):
        training_data, dev_data, test_data = {}, {}, {}
        dictionaries = (training_data, dev_data, test_data)
        counter = 0
        for file in self.input_files:
            input_file = open(os.path.join(sys.path[0], file), 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")
                result = []
                morph = ''
                tag = False

                for char in content[1]:
                    # Surface Segmentation
                    if char == '-':
                        result.append(morph)
                        morph = ''
                    else:
                        morph += char

                if morph.strip():
                    result.append(morph.strip())

                label = ''
                for morph in result:
                    if len(morph) == 1:
                        label += 'S'
                    else:
                        label += 'B'
                        for i in range(len(morph) - 2):
                            label += 'M'
                        label += 'E'

                dictionaries[counter][content[0]] = label

            counter += 1

        best_max_iteration = 160
        best_epsilon = 1e-07

        X_training, Y_training, words_training = surface_segment_data_preparation(training_data)
        X_dev, Y_dev, words_dev = surface_segment_data_preparation(dev_data)
        crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=best_epsilon, max_iterations=best_max_iteration)
        crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)

        return crf

    def surface_labelled_segmentation(self):
        """
        This method makes use of sklearn crfsuite to perform segment labelling of correct segments
        :return: list of predicted labels and list of correct labels
        """
        tic = time.perf_counter()

        # Collect the data
        ###########################################
        training_data, dev_data, test_data = {}, {}, {}
        dictionaries = (training_data, dev_data, test_data)
        counter = 0
        for file in self.input_files:
            input_file = open(os.path.join(sys.path[0], file), 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")
                labels = '-'.join(get_labels(content[2]))
                segments = removeLabels(content[2])

                # dictionaries[counter][content[0]] = [segments, labels] # word:[[segments],[labels]]
                dictionaries[counter][segments] = labels  # segments : labels
            input_file.close()
            counter += 1

        toc = time.perf_counter()
        print("Data Collected in " + str(tic - toc.__round__(2)))

        # Evaluate Model On the Test Set Using Optimised Model
        #######################################################

        best_delta = 8
        best_epsilon = 0.0000001
        best_max_iteration = 160
        best_algo = 'ap'

        best_epsilon, best_max_iteration = 0, 0
        maxF1 = 0
        print("Beginning Feature Computation and Model Optimisation")
        tic = time.perf_counter()

        '''for epsilon in [0.001, 0.00001, 0.0000001]:
            for max_iterations in [80, 120, 160, 200]:
                X_training, Y_training, words_training = surface_labelled_data_preparation(training_data)
                X_dev, Y_dev, words_dev = surface_labelled_data_preparation(dev_data)
                crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=epsilon, max_iterations=max_iterations)
                crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)

                Y_predict = crf.predict(X_dev)
                # f1 = f1_score(Y_dev, Y_predict, average='micro')
                labels = list(crf.classes_)
                sorted_labels = sorted(labels)
                f1 = metrics.flat_f1_score(Y_dev, Y_predict, average='micro', labels=labels, zero_division=0)
                if f1 > maxF1:
                    f1 = maxF1
                    best_epsilon = epsilon
                    best_max_iteration = max_iterations

        print(best_max_iteration)
        print(best_epsilon)'''

        X_training, Y_training, words_training = surface_labelled_data_preparation(training_data)
        X_dev, Y_dev, words_dev = surface_labelled_data_preparation(dev_data)
        X_test, Y_test, words_test = surface_labelled_data_preparation(test_data)
        print("Data Processed")

        best_epsilon = 1e-07
        best_max_iteration = 280
        best_algo = 'ap'

        # crf = sklearn_crfsuite.CRF(algorithm=best_algo, epsilon=best_epsilon, max_iterations=best_max_iteration)
        '''crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )'''
        crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=best_epsilon, max_iterations=best_max_iteration)
        print("CRF Initialized")
        '''print(len(X_training))
        counter = 0
        for feat in X_training:
            if len(feat) != len(Y_training[counter]):
                print(counter)
                print(len(feat))
                print(feat)
                print(len(Y_training[counter]))
                print(Y_training[counter])
            counter += 1
            print("#####################################")

        print(len(Y_training))'''
        # crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)
        crf.fit(X_training, Y_training)
        print("Data Fitted")
        Y_predict = crf.predict(X_test)
        # print(Y_predict[0])
        # print(Y_test[0])
        labels = list(crf.classes_)
        sorted_labels = sorted(labels)
        return Y_predict, Y_test, sorted_labels

    def to_use_labelled_crf(self):
        training_data, dev_data, test_data = {}, {}, {}
        dictionaries = (training_data, dev_data, test_data)
        counter = 0
        for file in self.input_files:
            input_file = open(os.path.join(sys.path[0], file), 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")
                labels = '-'.join(get_labels(content[2]))
                segments = removeLabels(content[2])

                dictionaries[counter][segments] = labels  # segments : labels
            input_file.close()
            counter += 1

        X_training, Y_training, words_training = surface_labelled_data_preparation(training_data)

        best_epsilon = 1e-07
        best_max_iteration = 280

        crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=best_epsilon, max_iterations=best_max_iteration)

        crf.fit(X_training, Y_training)
        return crf

    def __surface_labelled_segmentation_pipeline(self, features):
        """
        This method makes use of sklearn crfsuite to perform segment labelling of predicted segments
        :param features: features of the predicted segments
        :return: list of predicted labels and list of correct labels
        """
        tic = time.perf_counter()

        # Collect the data
        ###########################################
        training_data, dev_data, test_data = {}, {}, {}
        dictionaries = (training_data, dev_data, test_data)
        counter = 0
        for file in self.input_files:
            input_file = open(os.path.join(sys.path[0], file), 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")
                labels = '-'.join(get_labels(content[2]))
                segments = removeLabels(content[2])

                # dictionaries[counter][content[0]] = [segments, labels] # word:[[segments],[labels]]
                dictionaries[counter][segments] = labels  # segments : labels
            input_file.close()
            counter += 1

        toc = time.perf_counter()
        print("Data Collected in " + str(tic - toc.__round__(2)))

        # Evaluate Model On the Test Set Using Optimised Model
        #######################################################

        print("Beginning Feature Computation and Model Optimisation")
        tic = time.perf_counter()

        X_training, Y_training, words_training = surface_labelled_data_preparation(training_data)
        X_dev, Y_dev, words_dev = surface_labelled_data_preparation(dev_data)
        X_test, Y_test, words_test = surface_labelled_data_preparation(test_data)
        print("Data Processed")

        best_epsilon = 1e-07
        best_max_iteration = 280
        best_algo = 'ap'

        # crf = sklearn_crfsuite.CRF(algorithm=best_algo, epsilon=best_epsilon, max_iterations=best_max_iteration)
        '''crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )'''
        crf = sklearn_crfsuite.CRF(algorithm='ap', epsilon=best_epsilon, max_iterations=best_max_iteration)
        print("CRF Initialized")
        # crf.fit(X_training, Y_training, X_dev=X_dev, y_dev=Y_dev)
        crf.fit(X_training, Y_training)
        print("Data Fitted")
        Y_predict = crf.predict(features)
        labels = list(crf.classes_)
        sorted_labels = sorted(labels)
        return Y_predict, Y_test

    def pipeline(self):
        """
        This method makes use of sklearn crfsuite to perform segment labelling of predicted segments
        :return: list of predicted labels and list of correct labels
        """
        predicted, real = self.surface_segmentation()
        # print(predicted[0:10])
        # print(len(predicted))
        test_file = "../morphology/" + self.language + "/" + self.language + ".clean.test.conll"
        input_file = open(os.path.join(sys.path[0], test_file), 'r')
        segmented_words = []

        # Only one entry per word for dictionary

        words = []
        labels = []
        for line in input_file.readlines():
            tmp = line.rstrip('\n').split(" | ")[0]
            label_arr = line.rstrip('\n').split(" | ")[2]
            label_arr = get_labels(label_arr)
            if tmp not in words:
                words.append(tmp)
                labels.append(label_arr)

        segmented_words = []
        for word, label in zip(words, predicted):
            tmp = []
            for i in range(len(label)):
                if label[i] == "S" or label[i] == "E":
                    tmp.append(word[i])
                    tmp.append("-")
                else:
                    tmp.append(word[i])
            tmp = "".join(tmp).rstrip("-")
            segmented_words.append(tmp)

        features = surface_labelled_data_preparation_pipeline(segmented_words)
        predicted, test = self.__surface_labelled_segmentation_pipeline(features)
        return predicted, labels


def eval_morph_segments(predicted, target):
    """
    Method used to calculate precision, recall and f1 score particularly useful where corresponding inner lists
    may be of different lengths
    :param predicted: the list of predicted labels
    :param target: the list of actual values
    :return: precision, recall and f1 score
    """
    correct = 0.0
    for pred, targ in zip(predicted, target):
        for p in pred:
            if p in targ:
                correct += 1

    predicted_length = sum([len(pred) for pred in predicted])
    target_length = sum([len(targ) for targ in target])

    precision, recall = correct / predicted_length, correct / target_length
    f_score = 2 / (1 / precision + 1 / recall)
    return precision, recall, f_score


def surface_segment_data_preparation(word_dictionary: {str, str}):
    """
    This Method is used to generate features for the crf that is performing the surface segmentation
    :param word_dictionary: A word dictionary with the keys being the words and the value being the list of labels
    corresponding to each character
    :return: List of features, List of Correct Labels, The word as a list
    """
    X = []
    Y = []
    words = []
    for word in word_dictionary:
        word_list = []
        word_label_list = []
        for i in range(len(word)):
            gram_dict = {}
            gram_arr = []

            ### Unigram
            # gram_dict[word[i]] = 1
            gram_dict["uni_" + word[i]] = 1
            gram_arr.append(word[i])

            ### BIGRAM
            try:
                tmp = word[i - 1: i + 1]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 2:
                        gram_dict["bi_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue
            try:
                tmp = word[i: i + 2]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 2:
                        gram_dict["bi_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            ### TRIGRAM
            try:
                tmp = word[i - 1: i + 2]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 3:
                        gram_dict["tri_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            ##  FourGram
            try:
                tmp = word[i - 1: i + 3]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 4:
                        gram_dict["four_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            try:
                tmp = word[i - 2: i + 2]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 4:
                        gram_dict["four_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            ## FiveGram
            try:
                tmp = word[i - 2: i + 3]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 5:
                        gram_dict["five_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            ## SixGram
            try:
                tmp = word[i - 3: i + 3]
                if tmp:
                    if len(tmp) == 6:
                        # gram_dict[tmp] = 1
                        gram_dict["six_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            try:
                tmp = word[i - 2: i + 4]
                if tmp:
                    if len(tmp) == 6:
                        # gram_dict[tmp] = 1
                        gram_dict["six_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            if word[i] in 'aeiou':
                gram_dict["vowel"] = 1
            else:
                gram_dict["const"] = 1

            if word[i].isupper():
                gram_dict["upper"] = 1
            else:
                gram_dict["lower"] = 1

            word_list.append(gram_dict)
            word_label_list.append(word_dictionary[word][i])

        X.append(word_list)
        Y.append(word_label_list)
        words.append([char for char in word])
    return X, Y, words


def surface_segment_data_active_preparation(word_list: [str]):
    X = []
    for word in word_list:
        word_list = []
        for i in range(len(word)):
            gram_dict = {}
            gram_arr = []

            ### Unigram
            # gram_dict[word[i]] = 1
            gram_dict["uni_" + word[i]] = 1
            gram_arr.append(word[i])

            ### BIGRAM
            try:
                tmp = word[i - 1: i + 1]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 2:
                        gram_dict["bi_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue
            try:
                tmp = word[i: i + 2]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 2:
                        gram_dict["bi_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            ### TRIGRAM
            try:
                tmp = word[i - 1: i + 2]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 3:
                        gram_dict["tri_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            ##  FourGram
            try:
                tmp = word[i - 1: i + 3]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 4:
                        gram_dict["four_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            try:
                tmp = word[i - 2: i + 2]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 4:
                        gram_dict["four_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            ## FiveGram
            try:
                tmp = word[i - 2: i + 3]
                if tmp:
                    # gram_dict[tmp] = 1
                    if len(tmp) == 5:
                        gram_dict["five_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            ## SixGram
            try:
                tmp = word[i - 3: i + 3]
                if tmp:
                    if len(tmp) == 6:
                        # gram_dict[tmp] = 1
                        gram_dict["six_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            try:
                tmp = word[i - 2: i + 4]
                if tmp:
                    if len(tmp) == 6:
                        # gram_dict[tmp] = 1
                        gram_dict["six_" + tmp] = 1
                        gram_arr.append(tmp)
            except IndexError:
                continue

            if word[i] in 'aeiou':
                gram_dict["vowel"] = 1
            else:
                gram_dict["const"] = 1

            if word[i].isupper():
                gram_dict["upper"] = 1
            else:
                gram_dict["lower"] = 1

            word_list.append(gram_dict)

        X.append(word_list)
    return X


def surface_labelled_data_preparation(word_dictionary: {str, str}):
    """
    This Method is used to generate features for the crf that is performing the segment labelling
    :param word_dictionary: A word dictionary with the keys being the words and the value being the list of labels
    corresponding to each segment
    :return: List of features, List of Correct Labels, The list of segments
    """
    X = []
    Y = []
    words = []

    for word in word_dictionary:
        segments = word.split('-')
        labels = word_dictionary[word].split('-')
        segment_features = []
        for i in range(len(segments)):
            throw_away = 0
            features = {}

            segment_length = len(segments[i])
            features['length'] = segment_length

            features['segment.lower()'] = segments[i].lower()
            features['pos_in_word'] = i

            if segment_length % 2 == 0:
                features['even'] = 1
            else:
                features['odd'] = 1

            features['begin'] = segments[i][0]
            features['end'] = segments[i][len(segments[i]) - 1]

            try:
                features['prev_segment'] = segments[i - 1]
            except IndexError:
                throw_away += 1

            try:
                features['next_segment'] = segments[i + 1]
            except IndexError:
                throw_away += 1

            if segments[0].isupper():
                features['start_upper'] = 1
            else:
                features['start_lower'] = 1

            if segments[0] in 'aeiou':
                features['first_vowel'] = 1
            else:
                features['first_const'] = 1

            segment_features.append(features)
        words.append(segments)

        X.append(segment_features)
        Y.append(labels)
        words.append(word)

    return X, Y, words


def surface_labelled_data_preparation_pipeline(word_list: [str]):
    """
    This Method is used to generate features for the crf that is performing the pipeline segment labelling
    :param word_list: A list of words
    :return: List of features
    """
    X = []

    for word in word_list:
        segments = word.split('-')
        segment_features = []
        for i in range(len(segments)):
            features = {}

            segment_length = len(segments[i])
            features['length'] = segment_length

            features['segment.lower()'] = segments[i].lower()
            features['pos_in_word'] = i

            if segment_length % 2 == 0:
                features['even'] = 1
            else:
                features['odd'] = 1

            features['begin'] = segments[i][0]
            features['end'] = segments[i][len(segments[i]) - 1]

            try:
                features['prev_segment'] = segments[i - 1]
            except IndexError:
                continue
                # continue

            try:
                features['next_segment'] = segments[i + 1]
            except IndexError:
                continue

            if segments[0].isupper():
                features['start_upper'] = 1
            else:
                features['start_lower'] = 1

            if segments[0] in 'aeiou':
                features['first_vowel'] = 1
            else:
                features['first_const'] = 1

            segment_features.append(features)

        X.append(segment_features)

    return X


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
        if char == "(" or char == "[":
            last_seen_bracket.append(char)
            str2_arr.append("-")
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


def get_labels(orthographic: str):
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


def x_run_average_surface(num: int, language: str):
    """
    Method to perform the surface segmentation 'num' times to get average over all runs
    :param num: The number of times to run the model to get average scores
    :param language: The language the model should operate on
    :return: The average precision, recall and f1 scores across all num runs
    """
    recall, precision, f1 = [], [], []
    for i in range(num):
        CRF = BaselineCRF(language)
        x, y = CRF.surface_segmentation()

        test = MultiLabelBinarizer().fit_transform(y)
        predicted = MultiLabelBinarizer().fit_transform(x)

        recall.append(recall_score(test, predicted, average='micro'))
        precision.append(precision_score(test, predicted, average='micro'))
        f1.append(f1_score(test, predicted, average='micro'))

    recall = sum(recall) / len(recall)
    precision = sum(precision) / len(precision)
    f1 = sum(f1) / len(f1)
    return recall, precision, f1


def x_run_average_labelled(num: int, language: str):
    """
    Method to perform the segment labelling of correct segments 'num' times to get average over all runs
    :param num: The number of times to run the model to get average scores
    :param language: The language the model should operate on
    :return: The average precision, recall and f1 scores across all num runs
    """
    recall, precision, f1 = [], [], []
    for i in range(num):
        CRF = BaselineCRF(language)
        predict, test, labels = CRF.surface_labelled_segmentation()

        p, r, f = eval_morph_segments(predict, test)
        precision.append(p)
        recall.append(r)
        f1.append(f)

    recall = sum(recall) / len(recall)
    precision = sum(precision) / len(precision)
    f1 = sum(f1) / len(f1)
    return recall, precision, f1


def x_run_average_pipeline(num: int, language: str):
    """
    Method to perform the segment labelling of predicted segments 'num' times to get average over all runs
    :param num: The number of times to run the model to get average scores
    :param language: The language the model should operate on
    :return: The average precision, recall and f1 scores across all num runs
    """
    precision, recall, f1 = [], [], []
    for i in range(num):
        CRF = BaselineCRF(language)
        predicted, labels = CRF.pipeline()
        p, r, f = eval_morph_segments(predicted, labels)
        precision.append(p)
        recall.append(r)
        f1.append(f)

    recall = sum(recall) / len(recall)
    precision = sum(precision) / len(precision)
    f1 = sum(f1) / len(f1)
    return recall, precision, f1


def demonstration():
    # zulu, xhosa, swati, ndebele
    # Make all the CRFs
    print("Starting CRF Generation")
    ndebele_surface = BaselineCRF("ndebele").to_use_surface_crf()
    ndebele_labelled = BaselineCRF("ndebele").to_use_labelled_crf()
    print("isiNdebele Done")
    swati_surface = BaselineCRF("swati").to_use_surface_crf()
    swati_labelled = BaselineCRF("swati").to_use_labelled_crf()
    print("siSwati Done")
    xhosa_surface = BaselineCRF("xhosa").to_use_surface_crf()
    xhosa_labelled = BaselineCRF("xhosa").to_use_labelled_crf()
    print("isiXhosa Done")
    zulu_surface = BaselineCRF("zulu").to_use_surface_crf()
    zulu_labelled = BaselineCRF("zulu").to_use_labelled_crf()
    print("isiZulu Done")

    print("Saving Models")
    ndebeleFile = "isiNdebeleSurfaceModel.sav"
    swatiFile = "siSwatiSurfaceModel.sav"
    xhosaFile = "isiXhosaSurfaceModel.sav"
    zuluFile = "isiZuluSurfaceModel.sav"

    pickle.dump(ndebele_surface, open(ndebeleFile, 'wb'))
    pickle.dump(swati_surface, open(swatiFile, 'wb'))
    pickle.dump(xhosa_surface, open(xhosaFile, 'wb'))
    pickle.dump(zulu_surface, open(zuluFile, 'wb'))
    print("Models Saved")

    print("CRF Generation Completed")
    #########################################################
    word = ""
    languages = ["ndebele", "swati", "xhosa", "zulu"]
    while True:
        word = input("Enter a word: ").rstrip(" ").rstrip("\n")
        if word == "quit":
            exit(0)

        language = input("Enter a language: ").rstrip(" ").rstrip("\n")

        while language not in languages:
            print("Invalid Language Entered")
            language = input("Enter a language: ").rstrip(" ").rstrip("\n")

        ans = []
        features = surface_segment_data_active_preparation([word])
        if language == "ndebele":
            ans = ndebele_surface.predict(features)
        elif language == "swati":
            ans = swati_surface.predict(features)
        elif language == "xhosa":
            ans = xhosa_surface.predict(features)
        elif language == "zulu":
            ans = zulu_surface.predict(features)

        labels = ans[0]
        word_list = list(word)

        tmp = []
        print(labels)
        print(word_list)
        for word, label in zip(word_list, labels):
            for i in range(len(label)):
                if label[i] == "S" or label[i] == "E":
                    tmp.append(word[i])
                    tmp.append("-")
                else:
                    tmp.append(word[i])
        tmp = "".join(tmp).rstrip("-")

        print("Segmented Word: "+tmp)
        features = surface_labelled_data_preparation_pipeline([tmp])
        if language == "ndebele":
            ans = ndebele_labelled.predict(features)
        elif language == "swati":
            ans = swati_labelled.predict(features)
        elif language == "xhosa":
            ans = xhosa_labelled.predict(features)
        elif language == "zulu":
            ans = zulu_labelled.predict(features)

        labels = ans[0]
        print("Segment Labels: "+str(labels))

demonstration()



