from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import os

import sys

import re
from sklearn.manifold import TSNE


def load_embedding(dictionary_path: str, embedding_path: str) -> (dict, np.ndarray):
    with open(dictionary_path) as dictionary_file:
        dictionary = literal_eval(dictionary_file.read())
    embedding = np.loadtxt(embedding_path)
    return dictionary, embedding


def to_char_embedding(dictionary: dict, embedding: np.ndarray) -> (dict, np.ndarray):
    """
    Calculate a character embedding by averaging an existing word embedding, as shown in
    http://minimaxir.com/2017/04/char-embeddings/
    """
    vectors = {"UNK": (np.zeros(embedding[0].shape), 1)}
    for token, idx in dictionary.items():
        if token == "UNK":
            continue
        for char in token:
            if char in vectors:
                vectors[char] = (vectors[char][0] + embedding[idx],
                                 vectors[char][1] + 1)
            else:
                vectors[char] = (embedding[idx], 1)

    embedding_items = {c: v[0]/v[1] for c, v in vectors.items()}.items()
    char_dictionary = {e[0]: idx for idx, e in enumerate(embedding_items)}
    char_embedding = np.array(list(map(lambda ei: ei[1], embedding_items)))
    return char_dictionary, char_embedding


def save_char_embedding(dictionary: dict, embedding: np.ndarray, path: str):
    with open(os.path.join(path, "char_dictionary.txt"), "w") as embedding_file:
        embedding_file.write(str(dictionary))
    np.savetxt(os.path.join(path, "char_embeddings.txt"), embedding)


def get_color(char: str) -> str:
    if re.match("[A-ZÄÖÜ]", char) is not None:
        return "red"
    if re.match("[a-zäöüß]", char) is not None:
        return "blue"
    if re.match("[0-9]", char) is not None:
        return "green"
    return "yellow"


def plot(dictionary: dict, embedding: np.ndarray, path: str="/tmp/char_tsne.png"):
    print("Plotting ...")
    tsne = TSNE(perplexity=7, n_components=2, init='pca', n_iter=5000, method='exact')
    low_dim_embedding = tsne.fit_transform(embedding)
    reverse_dict = {v: k for k, v in dictionary.items()}
    labels = [reverse_dict[i] for i in range(len(dictionary))]
    colors = list(map(get_color, labels))
    plt.figure(figsize=(32, 32))
    for i, label in enumerate(labels):
        x, y = low_dim_embedding[i, :]
        plt.scatter(x, y, color=colors[i])
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(path)
    print("Plot saved to", path)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("dictionary_path embedding_path output_dir")
        exit(-1)

    dictionary, embedding = load_embedding(sys.argv[1], sys.argv[2])
    char_dictionary, char_embedding = to_char_embedding(dictionary, embedding)
    plot(char_dictionary, char_embedding)
    save_char_embedding(char_dictionary, char_embedding, sys.argv[3])
