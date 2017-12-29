#!/usr/bin/python3
import codecs
import sys
from ast import literal_eval

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

from repl import get_probabilities


class RandomForest:
    def __init__(self, dictionary_path: str, embedding_path: str, training_data_file: str, test_data_file: str,
                 n_estimators: int=120):
        print(locals())

        with open(dictionary_path) as dictionary_file:
            self.dictionary = literal_eval(dictionary_file.read())
        self.embedding = np.loadtxt(embedding_path)
        print("embedding shape", np.shape(self.embedding))

        self._input_size = np.shape(self.embedding)[1]

        self._db = self.get_db(training_data_file)
        self._TRAINING_SAMPLES = len(self._db["groundtruths"])
        self._num_outputs = np.max(self._db["groundtruths"]) + 1

        self._num_inputs = len(self._db["ngrams"][0]) - 1

        self._db_validate = self.get_db(test_data_file)

        self._classifier = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)

        print("determined parameters: num_inputs=%d, input_size=%d" % (self._num_inputs, self._input_size))

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
        db["groundtruths"] = list(map(np.argmax, raw_db["groundtruths"]))
        db["ngrams"], db["groundtruths"], db["ngrams_raw"] = shuffle(db["ngrams"], db["groundtruths"], raw_db["ngrams"])
        print("%s loaded, containing %d entries, class distribution: %s"
              % (path, len(db["groundtruths"]), str(np.sum(np.asarray(db["groundtruths"]), axis=0))))
        return db

    def get_all_data(self, db):
        batch = dict()
        batch["ngrams"] = list(map(lambda ws: np.concatenate(self.context_from_ngram(ws)), db["ngrams"]))
        batch["ngrams_raw"] = db["ngrams_raw"][:]
        batch["groundtruths"] = db["groundtruths"][:]
        return batch

    def train(self):
        training_data = self.get_all_data(self._db)
        self._classifier.fit(X=training_data["ngrams"], y=training_data["groundtruths"])

    def save_weights(self, output_path):
        print("TODO stub save_weights")
        # np.savetxt(output_path + "/W_fc1.txt", self.W_fc1.eval())
        # np.savetxt(output_path + "/b_fc1.txt", self.b_fc1.eval())
        # if self.use_hidden_layer:
        #     np.savetxt(output_path + "/b_fc2.txt", self.b_fc2.eval())
        #     np.savetxt(output_path + "/W_fc2.txt", self.W_fc2.eval())

    def get_suggestion(self, ngram):
        scores = self.get_score(ngram)
        if np.max(scores) > .5 + 1/(1+np.exp(-0.5))-0.5 and np.min(scores) < .5 - (1/(1+np.exp(-0.5))-0.5):
            return np.argmax(scores)
        else:
            return -1

    def get_score(self, ngram):
        scores = self._classifier.predict_proba([ngram])
        return scores[0]

    def context_from_ngram(self, ngram: np.ndarray) -> np.ndarray:
        middle = int(self._num_inputs / 2)
        return np.concatenate([ngram[:middle], ngram[middle+1:]])

    def validate(self, verbose=False):
        correct = list(np.zeros(self._num_outputs))
        incorrect = list(np.zeros(self._num_outputs))
        unclassified = list(np.zeros(self._num_outputs))
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        validation_data = self.get_all_data(self._db_validate)
        for ngram, raw_ngram, ground_truth in zip(validation_data["ngrams"], validation_data["ngrams_raw"], validation_data["groundtruths"]):
            suggestion = self.get_suggestion(ngram)
            if suggestion == -1:
                unclassified[ground_truth] = unclassified[ground_truth] + 1
                if verbose:
                    print("no decision:", " ".join(raw_ngram))
                tn = tn + 1
                fn = fn + 1
            elif suggestion == ground_truth:
                correct[ground_truth] = correct[ground_truth] + 1
                if verbose:
                    print("correct suggestion:", " ".join(raw_ngram))
                tp = tp + 1
                tn = tn + 1
            else:
                incorrect[ground_truth] = incorrect[ground_truth] + 1
                if verbose:
                    print("possible wrong suggestion:", " ".join(raw_ngram))
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

    def validate_error_detection(self, suggestion_threshold: float=0.5, error_threshold: float=0.2, verbose=False):
        correct = list(np.zeros(self._num_outputs))
        incorrect = list(np.zeros(self._num_outputs))
        unclassified = list(np.zeros(self._num_outputs))
        tp = 0
        fp = 0
        fn = 0

        validation_data = self.get_all_data(self._db_validate)
        for ngram, raw_ngram, ground_truth in zip(validation_data["ngrams"], validation_data["ngrams_raw"], validation_data["groundtruths"]):
            scores = self.get_score(ngram)
            probabilities = get_probabilities(scores)
            best_match = self.get_suggestion(ngram)
            best_match_score = scores[best_match]
            ground_truth_probability = probabilities[ground_truth]

            if best_match_score > suggestion_threshold and error_threshold > ground_truth_probability:
                # suggest alternative
                incorrect[ground_truth] = incorrect[ground_truth] + 1
                if verbose:
                    print("false alarm:", " ".join(raw_ngram))
                fp = fp + 1
                fn = fn + 1
            elif ground_truth_probability > suggestion_threshold:
                # ground truth will be suggested
                correct[ground_truth] = correct[ground_truth] + 1
                if verbose:
                    print("correct suggestion included:", " ".join(raw_ngram))
                tp = tp + 1
            else:
                # nothing happens
                unclassified[ground_truth] = unclassified[ground_truth] + 1
                if verbose:
                    print("no decision:", " ".join(raw_ngram))
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
        print("precision:", float(tp)/(tp+fp))
        print("recall:", float(tp)/(tp+fn))
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
    network = RandomForest(dictionary_path, embedding_path, training_data_file, test_data_file)
    network.train()
    network.save_weights(output_path)
    network.validate(verbose=True)
    # print(.5)
    # network.validate_error_detection(verbose=False, suggestion_threshold=.5, error_threshold=.5)
    # print(.4)
    # network.validate_error_detection(verbose=False, suggestion_threshold=.5, error_threshold=.4)
    # print(.3)
    # network.validate_error_detection(verbose=False, suggestion_threshold=.5, error_threshold=.3)
    # print(.2)
    # network.validate_error_detection(verbose=True, suggestion_threshold=.5, error_threshold=.2)


if __name__ == '__main__':
    main()
