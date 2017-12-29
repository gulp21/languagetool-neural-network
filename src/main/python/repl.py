import sys
from ast import literal_eval

import numpy as np

from LayeredScorer import LayeredScorer


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


def has_error(dictionary, embedding, scorer: LayeredScorer, ngram, subjects, suggestion_threshold=.5, error_threshold=.2) -> bool:
    """
    Parameters
    ----------
    suggestion_threshold:
        if the probability of another token is higher than this, it is considered as possible suggestion
    error_threshold:
        if the probability for the used token is less than this, it is considered wrong
    """
    middle = int(len(ngram) / 2)
    words = np.concatenate(list(map(lambda token: get_word_representation(dictionary, embedding, token), np.delete(ngram, middle))))
    scores = scorer.scores(words)
    probabilities = get_probabilities(scores)
    best_match_probability = probabilities[np.argmax(probabilities)]
    subject_index = subjects.index(ngram[middle])
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
    if len(sys.argv) != 4:
        raise ValueError("Expected dict, finalembedding, weights_path")

    dictionary_path = sys.argv[1]
    embedding_path = sys.argv[2]
    weights_path = sys.argv[3]

    with open(dictionary_path) as dictionary_file:
        dictionary = literal_eval(dictionary_file.read())
    embedding = np.loadtxt(embedding_path)

    subjects = ["als", "also", "da", "das", "dass", "de", "den", "denn", "die", "durch", "zur", "ihm", "im", "um", "nach", "noch", "war", "was"]
    # subjects = ["and", "end", "as", "at", "is", "do", "for", "four", "form", "from", "he", "if", "is", "its", "it", "no", "now", "on", "one", "same", "some", "than", "that", "then", "their", "there", "them", "the", "they", "to", "was", "way", "were", "where"]
    print(subjects)

    while True:
        ngram = input("ngram ").split(" ")
        try:
            scorer = LayeredScorer(weights_path)
            has_error(dictionary, embedding, scorer, ngram, subjects, error_threshold=.65, suggestion_threshold=.65)
        except ValueError as e:
            print(e)


if __name__ == '__main__':
    main()
