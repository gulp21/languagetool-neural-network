import sys
from ast import literal_eval
from itertools import takewhile

import numpy as np


def get_word_representation(dictionary, embedding, word):  # todo static
    if word in dictionary:
        return embedding[dictionary[word]]
    else:
        return embedding[dictionary["UNK"]]


def get_rating(scores, subjects):
    scored_suggestions = list(zip(scores, subjects))
    scored_suggestions.sort(key=lambda x: x[0], reverse=True)
    return scored_suggestions


def get_rating(scores, subjects):
    probabilities = get_probabilities(scores)
    scored_suggestions = list(zip(probabilities, scores, subjects))
    scored_suggestions.sort(key=lambda x: x[0], reverse=True)
    return scored_suggestions


def get_probabilities(scores):
    return 1 / (1 + np.exp(-np.array(scores)))


def check(dictionary, embedding, W, b, ngram, subjects, suggestion_threshold=.5, error_threshold=.2):
    words = np.concatenate(list(map(lambda token: get_word_representation(dictionary, embedding, token), np.delete(ngram, 2))))
    scores = words @ W + b
    best_match_score = scores[np.argmax(scores)]
    subject_index = subjects.index(ngram[2])
    subject_score = scores[subject_index]

    if best_match_score > suggestion_threshold and subject_score < error_threshold:
        print("ERROR detected, suggestions:", get_rating(scores, subjects))
    elif subject_score > suggestion_threshold:
        print("ok", get_rating(scores, subjects))
    else:
        print("no decision", get_rating(scores, subjects))


def main():
    if len(sys.argv) != 5:
        raise ValueError("Expected dict, finalembedding, W, b")

    dictionary_path = sys.argv[1]
    embedding_path = sys.argv[2]
    W_path = sys.argv[3]
    b_path = sys.argv[4]

    with open(dictionary_path) as dictionary_file:
        dictionary = literal_eval(dictionary_file.read())
    embedding = np.loadtxt(embedding_path)
    W = np.loadtxt(W_path)
    b = np.loadtxt(b_path)

    subjects = ["seid", "seit", "sei", "mein", "dein", "fein", "sein"]
    print(subjects)

    while True:
        ngram = input("5gram ").split(" ")
        check(dictionary, embedding, W, b, ngram, subjects)


if __name__ == '__main__':
    main()
