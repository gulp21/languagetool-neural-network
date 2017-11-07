#!/usr/bin/python3
import codecs
import math
import sys
from ast import literal_eval

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

import nn


class NeuralNetwork:
    def __init__(self, dictionary_path, embedding_path, training_data_file,
                 batch_size=1000, epochs=3000, use_hidden_layer=False, num_inputs=4, num_outputs=2):
        print(locals())

        self.use_hidden_layer = use_hidden_layer
        self.hidden_layer_size = 8
        self.batch_size = batch_size
        self.epochs = epochs

        with open(dictionary_path) as dictionary_file:
            self.dictionary = literal_eval(dictionary_file.read())
        self.embedding = np.loadtxt(embedding_path)
        print("embedding shape", np.shape(self.embedding))

        self._input_size = np.shape(self.embedding)[1]
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs

        self._db = self.get_db(training_data_file, oversample=True)
        self._TRAINING_SAMPLES = len(self._db["groundtruths"])
        self._current_batch_number = 0

        self.setup_net()

    def setup_net(self):
        with tf.name_scope('input'):
            self.words = []
            for i in range(self._num_inputs):
                self.words.append(tf.placeholder(tf.float32, [None, self._input_size]))
            x = tf.concat(self.words, 1)

        with tf.name_scope('ground-truth'):
            self.y_ = tf.placeholder(tf.float32, shape=[None, self._num_outputs])

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

    def get_db(self, path, oversample=False): #TODO oversample
        db = dict()
        raw_db = eval(codecs.open(path, "r", "utf-8").read())
        # raw_db = {'ngrams':[['too','early','to','rule','out'],['park','next','to','the','town'],['has','right','to','destroy','houses'],['ll','have','to','move','them'],['percent','increase','to','its','budget'],['ll','continue','to','improve','in'],['This','applies','to','footwear','too'],['t','appear','to','be','too'],['taking','action','to','prevent','a'],['too','close','to','the','fire'],['It','rates','to','get','a'],['it','was','too','early','to'],['houses',',','too','.','We'],['to','footwear','too','-','can'],['to','be','too','much','money'],['been','waiting','too','long','for'],['leave','rates','too','low','for'],['low','for','too','long',','],['not','going','too','close','to']],'groundtruths':[[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]}
        db["ngrams"] = np.asarray(
            list(map(lambda ws: list(map(lambda w: self.get_word_representation(w), ws)), raw_db["ngrams"])))
        db["groundtruths"] = raw_db["groundtruths"]
        db["ngrams"], db["groundtruths"] = shuffle(db["ngrams"], db["groundtruths"])
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
        self.assign_4gram_to_batch(batch, ngrams)
        batch[self.y_] = self._db["groundtruths"][start_index:end_index]
        self._current_batch_number = self._current_batch_number + 1
        # print("d" + str(len(batch[self.word1])))
        return batch

    def get_all_training_data(self):
        batch = dict()
        ngrams = self._db["ngrams"][:]
        self.assign_4gram_to_batch(batch, ngrams)
        batch[self.y_] = self._db["groundtruths"][:]
        # print("d" + str(len(batch[self.word1])))
        return batch

    def assign_4gram_to_batch(self, batch, ngrams):
        if self._num_inputs == 2:
            batch[self.words[0]] = ngrams[:, 1]
            batch[self.words[1]] = ngrams[:, 3]
        elif self._num_inputs == 4:
            batch[self.words[0]] = ngrams[:, 0]
            batch[self.words[1]] = ngrams[:, 1]
            batch[self.words[2]] = ngrams[:, 3]
            batch[self.words[3]] = ngrams[:, 4]

    def train(self):
        steps = math.ceil(self._TRAINING_SAMPLES / self.batch_size)
        print("Steps: %d, %d steps in %d epochs" % (steps * self.epochs, steps, self.epochs))
        for e in range(self.epochs):
            for i in range(steps):
                fd = self.get_batch()
                _ = self.sess.run([self.train_step], fd)  # train with next batch
            if e % 10 == 0:
                self._print_accuracy(e)
        self._print_accuracy(self.epochs)

    def _print_accuracy(self, epoch):
        train_acc = self.sess.run([self.accuracy], self.get_all_training_data())
        print("epoch %d, training accuracy %f" % (epoch, train_acc[0]))
        sys.stdout.flush()

    def save_weights(self, output_path):
        np.savetxt(output_path + "/W_fc1.txt", self.W_fc1.eval())
        np.savetxt(output_path + "/b_fc1.txt", self.b_fc1.eval())
        if self.use_hidden_layer:
            np.savetxt(output_path + "/b_fc2.txt", self.b_fc2.eval())
            np.savetxt(output_path + "/W_fc2.txt", self.W_fc2.eval())

    def get_suggestion(self, ngram):
        if self._num_inputs == 4:
            fd = {self.words[0]: [ngram[0]],
                  self.words[1]: [ngram[1]],
                  self.words[2]: [ngram[3]],
                  self.words[3]: [ngram[4]],
                  self.y_: [[0, 0]]}
        elif self._num_inputs == 2:
            fd = {self.words[0]: [ngram[1]],
                  self.words[1]: [ngram[3]],
                  self.y_: [[0, 0]]}
        scores = self.y.eval(fd)[0]
        if np.max(scores) > .5 and np.min(scores) < -.5:
            return np.argmax(scores)
        else:
            return -1

    def validate(self, test_data_file):
        db_validate = self.get_db(test_data_file, oversample=False)

        correct = [0, 0]
        incorrect = [0, 0]
        unclassified = [0, 0]
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for i in range(len(db_validate["groundtruths"])):
            suggestion = self.get_suggestion(db_validate["ngrams"][i])
            ground_truth = np.argmax(db_validate["groundtruths"][i])
            if suggestion == -1:
                unclassified[ground_truth] = unclassified[ground_truth] + 1
                tn = tn + 1
                fn = fn + 1
            elif suggestion == ground_truth:
                correct[ground_truth] = correct[ground_truth] + 1
                tp = tp + 1
                tn = tn + 1
            else:
                incorrect[ground_truth] = incorrect[ground_truth] + 1
                fp = fp + 1
                fn = fn + 1

        accuracy = list(map(lambda c, i: c/(c+i), correct, incorrect))
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
        print("precision:", float(tp)/(tp+fp))
        print("recall:", float(tp)/(tp+fn))


def main():
    if len(sys.argv) != 6:
        print("dictionary_path embedding_path training_data_file test_data_file output_path")
        exit(-1)
    dictionary_path = sys.argv[1]
    embedding_path = sys.argv[2]
    training_data_file = sys.argv[3]
    test_data_file = sys.argv[4]
    output_path = sys.argv[5]
    network = NeuralNetwork(dictionary_path, embedding_path, training_data_file)
    network.train()
    network.save_weights(output_path)
    network.validate(test_data_file)


if __name__ == '__main__':
    main()
