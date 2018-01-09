#!/usr/bin/python3
import codecs
import json
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
                 batch_size: int=1000, epochs: int=1000, use_hidden_layer: bool=False, use_after: bool=True):
        print(locals())

        self.hidden_layer_size = 8
        self.use_after = use_after
        self.num_conv_filters = 32
        self.max_sequence_length = 30
        self.batch_size = batch_size
        self.epochs = epochs

        with open(dictionary_path) as dictionary_file:
            self.dictionary = literal_eval(dictionary_file.read())
        self.embedding = np.loadtxt(embedding_path)
        print("embedding shape", np.shape(self.embedding))

        self._embedding_size = np.shape(self.embedding)[1]

        self._db = self.get_db(training_data_file)
        self._TRAINING_SAMPLES = len(self._db["groundTruths"])
        self._num_outputs = len(self._db["groundTruths"][0])
        self._current_batch_number = 0

        self._db_validate = self.get_db(test_data_file)

        self.setup_net()

        print("determined parameters: embedding_size=%d" % self._embedding_size)

    def setup_net(self):
        input_length = self.max_sequence_length * (self.use_after + 1)

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, input_length * self._embedding_size])

        with tf.name_scope('ground-truth'):
            self.y_ = tf.placeholder(tf.float32, shape=[None, self._num_outputs])

        with tf.name_scope('conv_layer'):
            x_image = tf.reshape(self.x, [-1, input_length, self._embedding_size, 1])
            filter_size = 5
            self.W_conv1 = nn.weight_variable([filter_size, self._embedding_size, 1, self.num_conv_filters])
            self.b_conv1 = nn.bias_variable([self.num_conv_filters])
            h_conv1 = tf.nn.relu(nn.conv2d(x_image, self.W_conv1, padding="VALID") + self.b_conv1)
            h_pool1 = nn.max_pool(h_conv1, input_length - filter_size + 1)
            h_pool1_flat = tf.reshape(h_pool1, [-1, self.num_conv_filters])

        with tf.name_scope('readout_layer'):
            self.W_fc2 = nn.weight_variable([self.num_conv_filters, self._num_outputs])
            self.b_fc2 = nn.bias_variable([self._num_outputs])
            self.y = tf.matmul(h_pool1_flat, self.W_fc2) + self.b_fc2

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

    def to_max_length_start(self, sentence: [str]):
        """keep first <=max_sequence_length words, fill rest with UNK"""
        return (sentence + ["UNK"] * self.max_sequence_length)[:self.max_sequence_length]

    def to_max_length_end(self, sentence: [str]):
        """keep last <=max_sequence_length words, fill rest with UNK"""
        return (["UNK"] * self.max_sequence_length + sentence)[-self.max_sequence_length:]

    def embed(self, words: [str]) -> np.ndarray:
        return np.array(list(map(lambda w: self.get_word_representation(w), words)))

    def get_db(self, path):
        db = dict()
        raw_db = json.load(open(path))
        db["tokensBefore"] = np.asarray(list(map(lambda ws: self.embed(self.to_max_length_end(ws)).flatten(), raw_db["tokensBefore"])))
        db["tokensAfter"] = np.asarray(list(map(lambda ws: self.embed(self.to_max_length_start(ws)).flatten(), raw_db["tokensAfter"])))
        db["groundTruths"] = raw_db["groundTruths"]
        db["context_str"] = list(map(lambda b, a: " ".join(self.to_max_length_end(b)) + " … " + " ".join(self.to_max_length_start(a)), raw_db["tokensBefore"], raw_db["tokensAfter"]))
        db["tokensBefore"], db["tokensAfter"], db["groundTruths"], db["context_str"] = \
            shuffle(db["tokensBefore"], db["tokensAfter"], db["groundTruths"], db["context_str"])
        print("%s loaded, containing %d entries, class distribution: %s"
              % (path, len(db["groundTruths"]), str(np.sum(np.asarray(db["groundTruths"]), axis=0))))
        return db

    def get_batch(self):
        if self._current_batch_number * self.batch_size > self._TRAINING_SAMPLES:
            self._current_batch_number = 0
        start_index = self._current_batch_number * self.batch_size
        end_index = (self._current_batch_number + 1) * self.batch_size
        batch = dict()
        tokens_before = self._db["tokensBefore"][start_index:end_index]
        tokens_after = self._db["tokensAfter"][start_index:end_index]
        self.assign_sentences_to_batch(batch, tokens_before, tokens_after)
        batch[self.y_] = self._db["groundTruths"][start_index:end_index]
        self._current_batch_number = self._current_batch_number + 1
        # print("d" + str(len(batch[self.word1])))
        return batch

    def get_all_training_data(self):
        batch = dict()
        tokens_before = self._db["tokensBefore"][:]
        tokens_after = self._db["tokensAfter"][:]
        self.assign_sentences_to_batch(batch, tokens_before, tokens_after)
        batch[self.y_] = self._db["groundTruths"][:]
        # print("d" + str(len(batch[self.word1])))
        return batch

    def assign_sentences_to_batch(self, batch, tokens_before, tokens_after):
        if self.use_after:
            batch[self.x] = np.concatenate([tokens_before, tokens_after], axis=1)
        else:
            batch[self.x] = tokens_before

    def train(self):
        steps = math.ceil(self._TRAINING_SAMPLES / self.batch_size)
        print("Steps: %d, %d steps in %d epochs" % (steps * self.epochs, steps, self.epochs))
        for e in range(self.epochs):
            for i in range(steps):
                fd = self.get_batch()
                _ = self.sess.run([self.train_step], fd)  # train with next batch
            if e % 10 == 0:
                self._print_accuracy(e)
            if e % 1000 == 0 and e > 0:
                self.validate()
                self.validate_error_detection()
        self._print_accuracy(self.epochs)

    def _print_accuracy(self, epoch):
        train_acc = self.sess.run([self.accuracy], self.get_all_training_data())
        print("epoch %d, training accuracy %f" % (epoch, train_acc[0]))
        sys.stdout.flush()

    def save_weights(self, output_path):
        nn.write_4dmat(output_path + "/W_conv1.txt", tf.transpose(self.W_conv1, (3, 0, 1, 2)).eval())
        nn.write_4dmat(output_path + "/b_conv1.txt", self.b_conv1.eval())
        np.savetxt(output_path + "/b_fc2.txt", self.b_fc2.eval())
        np.savetxt(output_path + "/W_fc2.txt", self.W_fc2.eval())

    def get_suggestion(self, tokens_before, tokens_after, threshold=.5):
        scores = self.get_score(tokens_before, tokens_after)
        if np.max(scores) > threshold and np.min(scores) < -threshold:
            return np.argmax(scores)
        else:
            return -1

    def get_score(self, tokens_before, tokens_after):
        fd = {self.y_: [list(np.zeros(self._num_outputs))],
              self.x: [np.concatenate([tokens_before, tokens_after])] if self.use_after else [tokens_before]}
        scores = self.y.eval(fd)[0]
        return scores

    def validate(self, verbose=False, threshold=.5):
        print("--- Validation of word prediction, threshold", threshold)

        correct = list(np.zeros(self._num_outputs))
        incorrect = list(np.zeros(self._num_outputs))
        unclassified = list(np.zeros(self._num_outputs))
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for i in range(len(self._db_validate["groundTruths"])):
            suggestion = self.get_suggestion(self._db_validate["tokensBefore"][i], self._db_validate["tokensAfter"][i], threshold=threshold)
            ground_truth = np.argmax(self._db_validate["groundTruths"][i])
            if suggestion == -1:
                unclassified[ground_truth] = unclassified[ground_truth] + 1
                if verbose:
                    print("no decision:", self._db_validate["context_str"][i])
                tn = tn + 1
                fn = fn + 1
            elif suggestion == ground_truth:
                correct[ground_truth] = correct[ground_truth] + 1
                if verbose:
                    print("correct suggestion:", self._db_validate["context_str"][i])
                tp = tp + 1
                tn = tn + 1
            else:
                incorrect[ground_truth] = incorrect[ground_truth] + 1
                if verbose:
                    print("possible wrong suggestion:", self._db_validate["context_str"][i])
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

        for i in range(len(self._db_validate["groundTruths"])):
            scores = self.get_score(self._db_validate["tokensBefore"][i], self._db_validate["tokensAfter"][i])
            probabilities = get_probabilities(scores)
            best_match = self.get_suggestion(self._db_validate["tokensBefore"][i], self._db_validate["tokensAfter"][i])
            best_match_score = scores[best_match]
            ground_truth = np.argmax(self._db_validate["groundTruths"][i])
            ground_truth_probability = probabilities[ground_truth]

            if best_match_score > suggestion_threshold and error_threshold > ground_truth_probability:
                # suggest alternative
                incorrect[ground_truth] = incorrect[ground_truth] + 1
                if verbose:
                    print("false alarm:", " ".join(self._db_validate["context_str"][i]))
                fp = fp + 1
                fn = fn + 1
            elif ground_truth_probability > suggestion_threshold:
                # ground truth will be suggested
                correct[ground_truth] = correct[ground_truth] + 1
                if verbose:
                    print("correct suggestion included:", " ".join(self._db_validate["context_str"][i]))
                tp = tp + 1
            else:
                # nothing happens
                unclassified[ground_truth] = unclassified[ground_truth] + 1
                if verbose:
                    print("no decision:", " ".join(self._db_validate["context_str"][i]))
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
