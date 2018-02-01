#!/usr/bin/python3
import codecs
import math
import sys
from ast import literal_eval

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

import nn
from repl import get_probabilities


class NeuralNetwork:
    def __init__(self, dictionary_path: str, embedding_path: str, training_data_file: str, test_data_file: str,
                 batch_size: int=1000, epochs: int=1000, use_hidden_layer: bool=False, use_conv_layer: bool=False):
        print(locals())

        self.use_hidden_layer = use_hidden_layer
        self.use_conv_layer = use_conv_layer
        self.hidden_layer_size = 32
        self.num_conv_filters = 32
        self.batch_size = batch_size
        self.epochs = epochs

        with open(dictionary_path) as dictionary_file:
            self.dictionary = literal_eval(dictionary_file.read())
        self.embedding = np.loadtxt(embedding_path)
        print("embedding shape", np.shape(self.embedding))

        self._input_size = np.shape(self.embedding)[1]

        self._db = self.get_db(training_data_file)
        self._TRAINING_SAMPLES = len(self._db["groundtruths"])
        self._num_outputs = len(self._db["groundtruths"][0])
        self._current_batch_number = 0

        self._num_inputs = len(self._db["ngrams"][0]) - 1

        self._db_validate = self.get_db(test_data_file)

        self.setup_net()

        print("determined parameters: num_inputs=%d, input_size=%d" % (self._num_inputs, self._input_size))

    def setup_net(self):
        with tf.name_scope('input'):
            self.words = []
            for i in range(self._num_inputs):
                self.words.append(tf.placeholder(tf.float32, [None, self._input_size]))
            x = tf.concat(self.words, 1)

        with tf.name_scope('ground-truth'):
            self.y_ = tf.placeholder(tf.float32, shape=[None, self._num_outputs])

        if self.use_conv_layer:
            with tf.name_scope('conv_layer'):
                pooling_positions = 3
                x_image = tf.reshape(x, [-1, self._input_size, self._num_inputs, 1])
                self.W_conv1 = nn.weight_variable([self._input_size, self._num_inputs - (pooling_positions - 1), 1, self.num_conv_filters])
                self.b_conv1 = nn.bias_variable([self.num_conv_filters])
                h_conv1 = tf.nn.relu(nn.conv2d(x_image, self.W_conv1) + self.b_conv1)
                h_pool1 = nn.max_pool(h_conv1, self._input_size, self._num_inputs - (pooling_positions - 1))
                h_pool1_flat = tf.reshape(h_pool1, [-1, self.num_conv_filters * pooling_positions])

        if self.use_hidden_layer:
            with tf.name_scope('hidden_layer'):
                self.W_fc1 = nn.weight_variable([self._num_inputs * self._input_size, self._num_inputs * self.hidden_layer_size])
                self.b_fc1 = nn.bias_variable([self._num_inputs * self.hidden_layer_size])
                hidden_layer = tf.nn.relu(tf.matmul(x, self.W_fc1) + self.b_fc1)

        with tf.name_scope('readout_layer'):
            if self.use_hidden_layer:
                self.W_fc2 = nn.weight_variable([self._num_inputs * self.hidden_layer_size, self._num_outputs])
                self.b_fc2 = nn.bias_variable([self._num_outputs])
                self.y = tf.matmul(hidden_layer, self.W_fc2) + self.b_fc2
            elif self.use_conv_layer:
                self.W_fc2 = nn.weight_variable([self.num_conv_filters * pooling_positions, self._num_outputs])
                self.b_fc2 = nn.bias_variable([self._num_outputs])
                self.y = tf.matmul(h_pool1_flat, self.W_fc2) + self.b_fc2
            else:
                self.W_fc1 = nn.weight_variable([self._num_inputs * self._input_size, self._num_outputs])
                self.b_fc1 = nn.bias_variable([self._num_outputs])
                self.y = tf.matmul(x, self.W_fc1) + self.b_fc1

        with tf.name_scope('train'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))
            tf.summary.scalar('loss_function', cross_entropy)
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def get_word_representation(self, word):
        if word in self.dictionary:
            return self.embedding[self.dictionary[word]]
        else:
            return self.embedding[self.dictionary["UNK"]]

    def get_db(self, path):
        db = dict()
        raw_db = eval(codecs.open(path, "r", "utf-8").read())
        # raw_db = {'ngrams':[['too','early','to','rule','out'],['park','next','to','the','town'],['has','right','to','destroy','houses'],['ll','have','to','move','them'],['percent','increase','to','its','budget'],['ll','continue','to','improve','in'],['This','applies','to','footwear','too'],['t','appear','to','be','too'],['taking','action','to','prevent','a'],['too','close','to','the','fire'],['It','rates','to','get','a'],['it','was','too','early','to'],['houses',',','too','.','We'],['to','footwear','too','-','can'],['to','be','too','much','money'],['been','waiting','too','long','for'],['leave','rates','too','low','for'],['low','for','too','long',','],['not','going','too','close','to']],'groundtruths':[[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]}
        db["ngrams"] = np.asarray(
            list(map(lambda ws: list(map(lambda w: self.get_word_representation(w), ws)), raw_db["ngrams"])))
        db["groundtruths"] = raw_db["groundtruths"]
        db["ngrams"], db["groundtruths"], db["ngrams_raw"] = shuffle(db["ngrams"], db["groundtruths"], raw_db["ngrams"])
        print("%s loaded, containing %d entries, class distribution: %s"
              % (path, len(db["groundtruths"]), str(np.sum(np.asarray(db["groundtruths"]), axis=0))))
        return db

    def get_batch(self):
        if self._current_batch_number * self.batch_size > self._TRAINING_SAMPLES:
            self._current_batch_number = 0
        start_index = self._current_batch_number * self.batch_size
        end_index = (self._current_batch_number + 1) * self.batch_size
        batch = dict()
        ngrams = self._db["ngrams"][start_index:end_index]
        self.assign_ngram_to_batch(batch, ngrams)
        batch[self.y_] = self._db["groundtruths"][start_index:end_index]
        self._current_batch_number = self._current_batch_number + 1
        # print("d" + str(len(batch[self.word1])))
        return batch

    def get_all_training_data(self):
        batch = dict()
        ngrams = self._db["ngrams"][:]
        self.assign_ngram_to_batch(batch, ngrams)
        batch[self.y_] = self._db["groundtruths"][:]
        # print("d" + str(len(batch[self.word1])))
        return batch

    def assign_ngram_to_batch(self, batch, ngrams):
        for idx, word in enumerate(self.words):
            i = self.context_index_for_ngram_position(idx)
            batch[word] = ngrams[:, i]

    def train(self):
        steps = math.ceil(self._TRAINING_SAMPLES / self.batch_size)
        print("Steps: %d, %d steps in %d epochs" % (steps * self.epochs, steps, self.epochs))
        for e in range(self.epochs):
            for i in range(steps):
                fd = self.get_batch()
                _ = self.sess.run([self.train_step], fd)  # train with next batch
            if e % 1 == 0:
                self._print_accuracy(e)
            if e % 1000 == 0:
                print("--- VALIDATION PERFORMANCE -")
                self.validate()
                self.validate_error_detection()
                print("----------------------------")
        self._print_accuracy(self.epochs)

    def _print_accuracy(self, epoch):
        train_acc = self.sess.run([self.accuracy], self.get_all_training_data())
        print("epoch %d, training accuracy %f" % (epoch, train_acc[0]))
        sys.stdout.flush()

    def save_weights(self, output_path):
        if self.use_conv_layer:
            nn.write_4dmat(output_path + "/W_conv1.txt", tf.transpose(self.W_conv1, (3, 0, 1, 2)).eval())
            nn.write_4dmat(output_path + "/b_conv1.txt", self.b_conv1.eval())
        else:
            np.savetxt(output_path + "/W_fc1.txt", self.W_fc1.eval())
            np.savetxt(output_path + "/b_fc1.txt", self.b_fc1.eval())
        if self.use_hidden_layer or self.use_conv_layer:
            np.savetxt(output_path + "/b_fc2.txt", self.b_fc2.eval())
            np.savetxt(output_path + "/W_fc2.txt", self.W_fc2.eval())

    def get_suggestion(self, ngram, threshold=.5):
        scores = self.get_score(ngram)
        if np.max(scores) > threshold and np.min(scores) < -threshold:
            return np.argmax(scores)
        else:
            return -1

    def get_score(self, ngram):
        fd = {self.y_: [list(np.zeros(self._num_outputs))]}
        for idx, word in enumerate(self.words):
            i = self.context_index_for_ngram_position(idx)
            fd[word] = [ngram[i]]
        scores = self.y.eval(fd)[0]
        return scores

    def context_index_for_ngram_position(self, idx: int) -> int:
        """for a position in a ngram, return idx if idx < floor(n/2), idx+1 otherwise"""
        if idx < int(self._num_inputs / 2):
            i = idx
        else:
            i = idx + 1
        return i

    def validate(self, verbose=False, threshold=.5):
        print("--- Validation of word prediction, threshold", threshold)

        correct = list(np.zeros(self._num_outputs))
        incorrect = list(np.zeros(self._num_outputs))
        unclassified = list(np.zeros(self._num_outputs))
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for i in range(len(self._db_validate["groundtruths"])):
            suggestion = self.get_suggestion(self._db_validate["ngrams"][i], threshold=threshold)
            ground_truth = np.argmax(self._db_validate["groundtruths"][i])
            if suggestion == -1:
                unclassified[ground_truth] = unclassified[ground_truth] + 1
                if verbose:
                    print("no decision:", " ".join(self._db_validate["ngrams_raw"][i]))
                tn = tn + 1
                fn = fn + 1
            elif suggestion == ground_truth:
                correct[ground_truth] = correct[ground_truth] + 1
                if verbose:
                    print("correct suggestion:", " ".join(self._db_validate["ngrams_raw"][i]))
                tp = tp + 1
                tn = tn + 1
            else:
                incorrect[ground_truth] = incorrect[ground_truth] + 1
                if verbose:
                    print("possible wrong suggestion:", " ".join(self._db_validate["ngrams_raw"][i]))
                fp = fp + 1
                fn = fn + 1

        accuracy = list(map(lambda c, i: c/(c+i) if (c+i) > 0 else np.nan, correct, incorrect))
        total_accuracy = list(map(lambda c, i, u: c/(c+i+u), correct, incorrect, unclassified))

        print("correct:", correct)
        print("incorrect:", incorrect)
        print("accuracy:", accuracy)
        print("unclassified:", unclassified)
        print("total accuracy:", total_accuracy)

        print("tp", tp)
        print("tn", tn)
        print("fp", fp)
        print("fn", fn)
        print("precision:", float(tp)/(tp+fp) if (tp+fp) > 0 else 1)
        print("recall:", float(tp)/(tp+fn) if (tp+fn) > 0 else 0)

    def validate_error_detection(self, suggestion_threshold: float=0.5, error_threshold: float=0.2, verbose=False):
        print("--- Error Detection Validation: suggestion_threshold %4.2f, error_threshold %4.2f"
              % (suggestion_threshold, error_threshold))

        correct = list(np.zeros(self._num_outputs))
        incorrect = list(np.zeros(self._num_outputs))
        unclassified = list(np.zeros(self._num_outputs))
        tp = 0
        fp = 0
        fn = 0

        for i in range(len(self._db_validate["groundtruths"])):
            scores = self.get_score(self._db_validate["ngrams"][i])
            probabilities = get_probabilities(scores)
            best_match = self.get_suggestion(self._db_validate["ngrams"][i])
            best_match_score = scores[best_match]
            ground_truth = np.argmax(self._db_validate["groundtruths"][i])
            ground_truth_probability = probabilities[ground_truth]

            if best_match_score > suggestion_threshold and error_threshold > ground_truth_probability:
                # suggest alternative
                incorrect[ground_truth] = incorrect[ground_truth] + 1
                if verbose:
                    print("false alarm:", " ".join(self._db_validate["ngrams_raw"][i]))
                fp = fp + 1
                fn = fn + 1
            elif ground_truth_probability > suggestion_threshold:
                # ground truth will be suggested
                correct[ground_truth] = correct[ground_truth] + 1
                if verbose:
                    print("correct suggestion included:", " ".join(self._db_validate["ngrams_raw"][i]))
                tp = tp + 1
            else:
                # nothing happens
                unclassified[ground_truth] = unclassified[ground_truth] + 1
                if verbose:
                    print("no decision:", " ".join(self._db_validate["ngrams_raw"][i]))
                fn = fn + 1

        accuracy = list(map(lambda c, i: c/(c+i), correct, incorrect))
        total_accuracy = list(map(lambda c, i, u: c/(c+i+u), correct, incorrect, unclassified))

        print("correct:", correct)
        print("incorrect:", incorrect)
        print("accuracy:", accuracy)
        print("unclassified:", unclassified)
        print("total accuracy:", total_accuracy)

        print("tp", tp)
        print("fp", fp)
        print("fn", fn)
        print("precision:", float(tp)/(tp+fp) if (tp+fp) > 0 else 1)
        print("recall:", float(tp)/(tp+fn) if (tp+fn) > 0 else 0)
        accuracy = list(map(lambda c, i: c/(c+i), correct, incorrect))
        print("accuracy:", accuracy)

        micro_accuracy = np.sum(correct)/(np.sum(correct)+np.sum(incorrect))
        print("micro accuracy:", micro_accuracy)


def main():
    if len(sys.argv) != 6:
        print("dictionary_path embedding_path training_data_file test_data_file output_path")
        exit(-1)
    dictionary_path = sys.argv[1]
    embedding_path = sys.argv[2]
    training_data_file = sys.argv[3]
    test_data_file = sys.argv[4]
    output_path = sys.argv[5]
    network = NeuralNetwork(dictionary_path, embedding_path, training_data_file, test_data_file)
    network.train()
    network.save_weights(output_path)
    network.validate(verbose=True, threshold=.5)
    network.validate(verbose=False, threshold=1)
    network.validate_error_detection(verbose=False, suggestion_threshold=.5, error_threshold=.5)
    network.validate_error_detection(verbose=False, suggestion_threshold=.5, error_threshold=.4)
    network.validate_error_detection(verbose=False, suggestion_threshold=.5, error_threshold=.3)
    network.validate_error_detection(verbose=False, suggestion_threshold=.5, error_threshold=.2)


if __name__ == '__main__':
    main()
