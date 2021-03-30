# from bilstm_crf import BiLSTM_CRF
# from .bi_lstm_crf.app import train, WordsTagger
import argparse
import os
import sys
import torch

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

from bi_lstm_crf.app import train, WordsTagger


class NeuralCRF:
    """
    Neural CRF implementation using edited version of Bi-LSTM-CRF
    """

    def __init__(self, language: str):
        """
        The constructor function for the Bi-LSTM-CRF that takes in the name of the language and looks for the file
        corresponding to that language in the morphology folder
        :param language: string of the language that particular model should focus on
        """
        self.language = language
        self.input_files = ["../morphology/" + language + '/' + language + ".clean.train.conll",
                            "../morphology/" + language + '/' + language + ".clean.dev.conll",
                            "../morphology/" + language + '/' + language + ".clean.test.conll"]
        self.labels = []

    def surface_segmentation(self):
        """
        This method performs surface segmentation
        :return: list of predicted segments and list of correct segments
        """
        # Collect Data
        ####################################################
        training_data, dev_data, test_data = [], [], []
        lists = (training_data, dev_data, test_data)
        datas_set = open(os.path.join(sys.path[0], 'corpus_dir/dataset.txt'), "w")
        counter = 0
        for file in self.input_files:

            input_file = open(os.path.join(sys.path[0], file), 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")
                word = list(content[0])
                segments = content[1].split('-')
                label = ""
                for morph in segments:
                    if len(morph) == 1:
                        label += "S"
                    else:
                        label += "B"
                        for i in range(len(morph) - 2):
                            label += "M"
                        label += "E"

                string = []
                for morph in segments:
                    if len(morph) == 1:
                        string.append("S")
                    else:
                        string.append("BW")
                        for i in range(len(morph) - 2):
                            string.append("M")
                        string.append("E")

                string = format_arrays_json(string)
                tmp = (word, label)
                lists[counter].append(tmp)
                if counter == 0:
                    datas_set.write("".join(word) + "\t" + string + "\n")
            input_file.close()
            counter += 1

        datas_set.close()
        print("Collected Data")

        vocab = open(os.path.join(sys.path[0], 'corpus_dir/vocab.json'), "w")
        tags = open(os.path.join(sys.path[0], 'corpus_dir/tags.json'), "w")

        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\'.\",1234567890%"
        string = "["
        count = 0
        for char in alphabet:
            if char == "\"":
                string += "\"\\" + char + "\""
            else:
                string += "\"" + char + "\""
            if count != len(alphabet) - 1:
                string += ", "
            count += 1
        string += "]"
        # vocab.write(str(list(alphabet)))
        vocab.write(string)
        vocab.close()

        word_to_ix = {}
        for letters in alphabet:
            word_to_ix[letters] = len(word_to_ix)
        tag_to_ix = {"S": 0, "B": 1, "M": 2, "E": 3}
        # tags.write("[\"S\", \"B\", \"M\", \"E\"]")
        tags.write("[\"S\", \"BW\", \"M\", \"E\"]")
        tags.close()

        embedding_dim = 5
        hidden_dim = 4

        ####################################################################################

        args = get_surface_args()
        print("Got Arguments")
        train(args)
        print("Completed Training")
        df = pd.read_csv("surface_model/loss.csv")
        df[["train_loss", "val_loss"]].ffill().plot(grid=True)
        plt.show()
        temp_predicted = []
        temp_true = []
        model = WordsTagger(model_dir='surface_model')

        print("Saving Model")
        modelName = self.language+"NeuralCRFSurfaceModel.sav"
        pickle.dump(model, open(modelName, 'wb'))
        print("Done Saving")


        print("Testing Model")
        for word, label in test_data:
            temp_true.append(label)
            tmp = model([''.join(word)], begin_tags="BS")[0]
            temp_predicted.append(tmp)

        y_true = []

        for lbl in temp_true:
            y_true.append(list(lbl))

        print(y_true[0:15])

        y_predicted = []
        for arr in temp_predicted:
            tmp = arr[0]
            str = []
            for char in tmp:
                if char == "BW":
                    str.append("B")
                else:
                    str.append(char)
            y_predicted.append(str)
        print(y_predicted[0:15])

        return y_predicted, y_true

    def surface_labelled_segmentation(self):
        """
        This method performs segment labelling of correct segments
        :return: list of predicted labels and list of correct labels
        """
        # Collect Data
        ####################################################
        training_data, dev_data, test_data = [], [], []
        lists = (training_data, dev_data, test_data)
        counter = 0
        alphabet, labels = [], []
        datas_set = open(os.path.join(sys.path[0], 'corpus_dir/dataset.txt'), "w")

        for file in self.input_files:
            input_file = open(os.path.join(sys.path[0], file), 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")

                surface_form = content[1].split('-')
                label_list = get_ortho_labels(content[2])

                for segment in surface_form:
                    if segment not in alphabet:
                        alphabet.append(segment)
                for label in label_list:
                    if label not in labels:
                        labels.append(label)

                tmp = (surface_form, label_list)
                lists[counter].append(tmp)

                tmp_surface = format_arrays_json(surface_form)

                tmp_labels = format_arrays_json(label_list)

                if counter == 0:
                    datas_set.write(tmp_surface + "\t" + tmp_labels + "\n")

            input_file.close()
            counter += 1

        datas_set.close()
        print("Collected Data: labels and segments")
        ##############################################
        vocab = open(os.path.join(sys.path[0], 'corpus_dir/vocab.json'), "w")
        tags = open(os.path.join(sys.path[0], 'corpus_dir/tags.json'), "w")

        tmp_alphabet = format_arrays_json(alphabet)
        tmp_labels = format_arrays_json(labels)
        self.labels = labels

        vocab.write(tmp_alphabet)
        vocab.close()

        tags.write(tmp_labels)
        tags.close()

        args = get_labelled_args()
        print("Got Arguments")

        train(args)
        print("Completed Training")

        df = pd.read_csv("labelled_model/loss.csv")
        df[["train_loss", "val_loss"]].ffill().plot(grid=True)
        plt.show()

        model = WordsTagger(model_dir='labelled_model/')
        print("Testing Model")

        y_true = []
        y_predicted = []
        for segments, labels in test_data:
            y_true.append(labels)

            tmp = model([segments])[0][0]
            y_predicted.append(tmp)

        return y_predicted, y_true

    def get_segments(self):
        """
        Method created to minimise number of computations happening at once
        """
        predicted, true = self.surface_segmentation()
        tmp_set = open(os.path.join(sys.path[0], 'corpus_dir/' + self.language + '_ segments.txt'), "w")
        for arr in predicted:
            string = ""
            for char in arr:
                string += char
            tmp_set.write(string + "\n")

    def pipeline(self):
        """
        Model that does the pipeline operation by reading the segments from the file to minimise computations per run
        """
        words = []
        test_file = "../morphology/" + self.language + '/' + self.language + ".clean.test.conll"
        tmp_set = 'corpus_dir/' + self.language + '_ segments.txt'
        predicted = []
        file = open(os.path.join(sys.path[0], tmp_set), 'r')
        for line in file.readlines():
            predicted.append([char for char in line.rstrip('\n')])

        # print(predicted)
        file = open(test_file, 'r')
        for line in file.readlines():
            words.append(line.rstrip('\n').split(" | ")[0])

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
            segmented_words.append(tmp.split("-"))

        training_data, dev_data, test_data = [], [], []
        lists = (training_data, dev_data, test_data)
        counter = 0
        alphabet, labels = [], []
        datas_set = open(os.path.join(sys.path[0], 'corpus_dir/dataset.txt'), 'w')

        for file in self.input_files:
            input_file = open(file, 'r')
            for line in input_file.readlines():
                content = line.rstrip('\n').split(" | ")

                surface_form = content[1].split('-')
                label_list = get_ortho_labels(content[2])

                for segment in surface_form:
                    if segment not in alphabet:
                        alphabet.append(segment)
                    for label in label_list:
                        if label not in labels:
                            labels.append(label)

                tmp = (surface_form, label_list)
                lists[counter].append(tmp)

                tmp_surface = format_arrays_json(surface_form)

                tmp_labels = format_arrays_json(label_list)

                if counter < 3:
                    datas_set.write(tmp_surface + "\t" + tmp_labels + "\n")

            input_file.close()
            counter += 1

            datas_set.close()
            print("Collected Data: labels and segments")
            ##############################################
            vocab = open(os.path.join(sys.path[0], 'corpus_dir/vocab.json'), "w")
            tags = open(os.path.join(sys.path[0], 'corpus_dir/tags.json'), "w")

            for arr in segmented_words:
                for segments in arr:
                    if segments not in labels:
                        labels.append(segments)

            tmp_alphabet = format_arrays_json(alphabet)
            print("Alphabet: " + str(len(alphabet)))
            # print(tmp_alphabet)
            tmp_labels = format_arrays_json(labels)
            print("Labels: " + str(len(labels)))
            # print(tmp_labels)
            self.labels = labels

            vocab.write(tmp_alphabet)
            vocab.close()

            tags.write(tmp_labels)
            tags.close()

            best_epoch = 10
            best_learning_rate = 0.0004
            best_weightd = 0
            best_rnn = 12
            # best_batches = 1000
            best_batches = 256
            args = get_labelled_args(best_epoch, best_learning_rate, best_weightd, best_rnn, best_batches)
            print("Got Arguments")

            train(args)
            print("Completed Training")

            model = WordsTagger(model_dir='labelled_model/')
            print("Testing Model")

            y_true = []
            y_predicted = []
            for i in range(len(segmented_words)):
                y_true.append(test_data[i][1])
                y_predicted.append(model([segmented_words[i]])[0][0])

            return y_predicted, y_true


def x_run_average_surface(num: int, language: str):
    """
    Method to perform the surface segmentation 'num' times to get average over all runs
    :param num: The number of times to run the model to get average scores
    :param language: The language the model should operate on
    :return: The average precision, recall and f1 scores across all num runs
    """
    recall, precision, f1 = [], [], []
    for i in range(num):
        n = NeuralCRF(language)
        x, y = n.surface_segmentation()

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
        n = NeuralCRF(language)
        predict, test = n.surface_labelled_segmentation()
        labels = n.labels

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
    recall, precision, f1 = [], [], []
    for i in range(num):
        n = NeuralCRF(language)
        predict, test = n.pipeline()

        p, r, f = eval_morph_segments(predict, test)
        precision.append(p)
        recall.append(r)
        f1.append(f)

    recall = sum(recall) / len(recall)
    precision = sum(precision) / len(precision)
    f1 = sum(f1) / len(f1)
    return recall, precision, f1


def eval_morph_segments(predicted, target):
    """
    Method used to calculate precision, recall and f1 score particularly useful where corresponding inner lists may be of different lengths
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


def get_surface_segments(surface: str):
    """
    Method to extract the segments from the orthographic form of the word
    :param orthographic: the orthographic form of the word
    :return: list of all the segments in the word
    """
    segments = []
    tmp = ''
    label = False

    # Get all segments from orthographic form
    for char in surface:
        if char == '[':
            segments.append(tmp)
            tmp = ''
            label = True
        elif char == ']':
            label = False
        elif not label:
            tmp += char
    return segments


def format_arrays_json(arr: [str], special_case=None):
    """
    A class to format an array in the JSON format needed by the class
    :param arr: the array of strings in the library or vocab
    :param special_case: included for special cases where there are " characters in front of the word
    :return: a string in the format required by the JSON file
    """
    string = "["
    count = 0
    for seg in arr:
        if seg == special_case:
            string += "\"\\" + seg + "\""
        else:
            string += "\"" + seg + "\""
        if count != len(arr) - 1:
            string += ", "
        count += 1
    string += "]"
    return string


def get_surface_args():
    """
    Method to get the arguments for the surface segmentation Bi_LSTM-CRF
    :return: The arguments for the surface segmentation Bi_LSTM-CRF
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('corpus_dir', type=str, help="the corpus directory")
    parser.add_argument('--corpus_dir', type=str, default='corpus_dir/', help="the corpus directory")
    # parser.add_argument('--model_dir', type=str, default="Neural CRF Model/surface_model", help="the output directory for model files")
    parser.add_argument('--model_dir', type=str, default="surface_model/",
                        help="the output directory for model files")

    parser.add_argument('--num_epoch', type=int, default=20, help="number of epoch to train")
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # parser.add_argument('--weight_decay', type=float, default=0., help='the L2 normalization parameter')
    parser.add_argument('--weight_decay', type=float, default=0., help='the L2 normalization parameter')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size for training')
    parser.add_argument('--device', type=str, default=None,
                        help='the training device: "cuda:0", "cpu:0". It will be auto-detected by default')
    parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length within training')

    # parser.add_argument('--val_split', type=float, default=0.2, help='the split for the validation dataset')
    parser.add_argument('--val_split', type=float, default=0.2, help='the split for the validation dataset')
    # parser.add_argument('--test_split', type=float, default=0.2, help='the split for the testing dataset')
    parser.add_argument('--test_split', type=float, default=0.2, help='the split for the testing dataset')
    parser.add_argument('--recovery', action="store_true",
                        help="continue to train from the saved model in model_dir")
    parser.add_argument('--save_best_val_model', action="store_true",
                        help="save the model whose validation score is smallest")

    parser.add_argument('--embedding_dim', type=int, default=100, help='the dimension of the embedding layer')
    parser.add_argument('--hidden_dim', type=int, default=128, help='the dimension of the RNN hidden state')
    parser.add_argument('--num_rnn_layers', type=int, default=1, help='the number of RNN layers')
    parser.add_argument('--rnn_type', type=str, default="lstm", help='RNN type, choice: "lstm", "gru"')

    args = parser.parse_args("")
    return args


def get_surface_args(epoch=None, learning_rate=None, weight_decay=None, rnn_layers=None, best_batches=None):
    """
    Method to get the arguments for the surface segmentation
    :param epoch: Value fornumber of epochs
    :param learning_rate: Value for learning rate
    :param weight_decay: value for weight decay
    :param rnn_layers: Value for number of rnn layers
    :param batches: Value for batches
    :return: The arguments for the labelling Bi_LSTM-CRF
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('corpus_dir', type=str, help="the corpus directory")

    parser.add_argument('--corpus_dir', type=str, default="corpus_dir/", help="the corpus directory")
    # parser.add_argument('--corpus_dir', type=str, default='/content/drive/My Drive/Colab Notebooks/CRF/corpus_dir/',
    #                    help="the corpus directory")

    parser.add_argument('--model_dir', type=str, default="model_dir/", help="the output directory for model files")
    #parser.add_argument('--model_dir', type=str, default="/content/drive/My Drive/Colab Notebooks/CRF/model_dir/",
    #                    help="the output directory for model files")

    parser.add_argument('--num_epoch', type=int, default=epoch, help="number of epoch to train")
    parser.add_argument('--lr', type=float, default=learning_rate, help='learning rate')
    # parser.add_argument('--weight_decay', type=float, default=0., help='the L2 normalization parameter')
    parser.add_argument('--weight_decay', type=float, default=weight_decay, help='the L2 normalization parameter')
    # parser.add_argument('--batch_size', type=int, default=1000, help='batch size for training')
    parser.add_argument('--batch_size', type=int, default=best_batches, help='batch size for training')
    parser.add_argument('--device', type=str, default=None,
                        help='the training device: "cuda:0", "cpu:0". It will be auto-detected by default')
    parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length within training')

    # parser.add_argument('--val_split', type=float, default=0.2, help='the split for the validation dataset')
    parser.add_argument('--val_split', type=float, default=0.2, help='the split for the validation dataset')
    # parser.add_argument('--test_split', type=float, default=0.2, help='the split for the testing dataset')
    parser.add_argument('--test_split', type=float, default=0.2, help='the split for the testing dataset')
    parser.add_argument('--recovery', action="store_true",
                        help="continue to train from the saved model in model_dir")
    parser.add_argument('--save_best_val_model', action="store_true",
                        help="save the model whose validation score is smallest")

    parser.add_argument('--embedding_dim', type=int, default=100, help='the dimension of the embedding layer')
    parser.add_argument('--hidden_dim', type=int, default=128, help='the dimension of the RNN hidden state')
    parser.add_argument('--num_rnn_layers', type=int, default=rnn_layers, help='the number of RNN layers')
    parser.add_argument('--rnn_type', type=str, default="lstm", help='RNN type, choice: "lstm", "gru"')

    args = parser.parse_args("")
    return args


def get_labelled_args():
    """
    Method to get the arguments for the labelling
    :return: The arguments for the labelling Bi_LSTM-CRF
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('corpus_dir', type=str, help="the corpus directory")
    parser.add_argument('--corpus_dir', type=str, default='corpus_dir/', help="the corpus directory")

    parser.add_argument('--model_dir', type=str, default="labelled_model/", help="the output directory for model files")

    parser.add_argument('--num_epoch', type=int, default=20, help="number of epoch to train")
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # parser.add_argument('--weight_decay', type=float, default=0., help='the L2 normalization parameter')
    parser.add_argument('--weight_decay', type=float, default=0., help='the L2 normalization parameter')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size for training')
    parser.add_argument('--device', type=str, default=None,
                        help='the training device: "cuda:0", "cpu:0". It will be auto-detected by default')
    parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length within training')

    # parser.add_argument('--val_split', type=float, default=0.2, help='the split for the validation dataset')
    parser.add_argument('--val_split', type=float, default=0.2, help='the split for the validation dataset')
    # parser.add_argument('--test_split', type=float, default=0.2, help='the split for the testing dataset')
    parser.add_argument('--test_split', type=float, default=0.2, help='the split for the testing dataset')
    parser.add_argument('--recovery', action="store_true",
                        help="continue to train from the saved model in model_dir")
    parser.add_argument('--save_best_val_model', action="store_true",
                        help="save the model whose validation score is smallest")

    parser.add_argument('--embedding_dim', type=int, default=100, help='the dimension of the embedding layer')
    parser.add_argument('--hidden_dim', type=int, default=128, help='the dimension of the RNN hidden state')
    parser.add_argument('--num_rnn_layers', type=int, default=1, help='the number of RNN layers')
    parser.add_argument('--rnn_type', type=str, default="lstm", help='RNN type, choice: "lstm", "gru"')

    args = parser.parse_args("")
    return args


def get_labelled_args(epoch=None, learning_rate=None, weight_decay=None, rnn_layers=None, batches=None):
    """
    Method to get the arguments for the labelling
    :param epoch: Value fornumber of epochs
    :param learning_rate: Value for learning rate
    :param weight_decay: value for weight decay
    :param rnn_layers: Value for number of rnn layers
    :param batches: Value for batches
    :return: The arguments for the labelling Bi_LSTM-CRF
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('corpus_dir', type=str, help="the corpus directory")
    parser.add_argument('--corpus_dir', type=str, default='/content/drive/My Drive/Colab Notebooks/CRF/corpus_dir/',
                        help="the corpus directory")

    parser.add_argument('--model_dir', type=str, default="/content/drive/My Drive/Colab Notebooks/CRF/model_dir/",
                        help="the output directory for model files")

    parser.add_argument('--num_epoch', type=int, default=epoch, help="number of epoch to train")
    parser.add_argument('--lr', type=float, default=learning_rate, help='learning rate')
    # parser.add_argument('--weight_decay', type=float, default=0., help='the L2 normalization parameter')
    parser.add_argument('--weight_decay', type=float, default=weight_decay, help='the L2 normalization parameter')
    # parser.add_argument('--batch_size', type=int, default=1000, help='batch size for training')
    parser.add_argument('--batch_size', type=int, default=batches, help='batch size for training')
    parser.add_argument('--device', type=str, default=None,
                        help='the training device: "cuda:0", "cpu:0". It will be auto-detected by default')
    parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length within training')

    # parser.add_argument('--val_split', type=float, default=0.2, help='the split for the validation dataset')
    parser.add_argument('--val_split', type=float, default=0.2, help='the split for the validation dataset')
    # parser.add_argument('--test_split', type=float, default=0.2, help='the split for the testing dataset')
    parser.add_argument('--test_split', type=float, default=0.2, help='the split for the testing dataset')
    parser.add_argument('--recovery', action="store_true",
                        help="continue to train from the saved model in model_dir")
    parser.add_argument('--save_best_val_model', action="store_true",
                        help="save the model whose validation score is smallest")

    parser.add_argument('--embedding_dim', type=int, default=100, help='the dimension of the embedding layer')
    parser.add_argument('--hidden_dim', type=int, default=128, help='the dimension of the RNN hidden state')
    parser.add_argument('--num_rnn_layers', type=int, default=rnn_layers, help='the number of RNN layers')
    parser.add_argument('--rnn_type', type=str, default="lstm", help='RNN type, choice: "lstm", "gru"')

    args = parser.parse_args("")
    return args


def demonstration():
    languages = ["ndebele", "swati", "xhosa", "zulu"]
    for lang in languages:
        n = NeuralCRF(lang)
        x, y = n.surface_segmentation()

demonstration()
