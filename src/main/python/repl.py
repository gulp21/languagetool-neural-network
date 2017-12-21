import sys
from ast import literal_eval

import numpy as np


def get_word_representation(dictionary, embedding, word):  # todo static
    if word in dictionary:
        return embedding[dictionary[word]]
    else:
        print(" " + word + " is unknown")
        return embedding[dictionary["UNK"]]


def get_rating(scores, subjects):
    probabilities = get_probabilities(scores)
    scored_suggestions = list(zip(probabilities, scores, subjects))
    scored_suggestions.sort(key=lambda x: x[0], reverse=True)
    return scored_suggestions


def get_probabilities(scores):
    return 1 / (1 + np.exp(-np.array(scores)))


def has_error(dictionary, embedding, W, b, ngram, subjects, suggestion_threshold=.5, error_threshold=.2) -> bool:
    """
    Parameters
    ----------
    suggestion_threshold:
        if the probability of another token is higher than this, it is considered as possible suggestion
    error_threshold:
        if the probability for the used token is less than this, it is considered wrong
    """
    words = np.concatenate(list(map(lambda token: get_word_representation(dictionary, embedding, token), np.delete(ngram, 2))))
    scores = words @ W + b
    probabilities = get_probabilities(scores)
    best_match_probability = probabilities[np.argmax(probabilities)]
    subject_index = subjects.index(ngram[2])
    subject_probability = probabilities[subject_index]

    print("checked", ngram)

    if best_match_probability > suggestion_threshold and subject_probability < error_threshold:
        print("ERROR detected, suggestions:", get_rating(scores, subjects))
        return True
    elif subject_probability > suggestion_threshold:
        print("ok", get_rating(scores, subjects))
        return False
    else:
        print("no decision", get_rating(scores, subjects))
        return False


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

    subjects = ["als", "also", "da", "das", "dass", "de", "den", "denn", "die", "durch", "zur", "ihm", "im", "um", "nach", "noch", "war", "was"]
    print(subjects)

    while True:
        ngram = input("5gram ").split(" ")
        has_error(dictionary, embedding, W, b, ngram, subjects)


if __name__ == '__main__':
    main()
