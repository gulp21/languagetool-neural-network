import numpy as np


class LayeredScorer:

    def __init__(self, weights_path: str):
        self.W1 = np.loadtxt(weights_path + "W_fc1.txt")
        self.b1 = np.loadtxt(weights_path + "b_fc1.txt")
        try:
            self.W2 = np.loadtxt(weights_path + "W_fc2.txt")
            self.b2 = np.loadtxt(weights_path + "b_fc2.txt")
        except FileNotFoundError:
            self.W2 = None
            self.b2 = None

    def scores(self, m: np.ndarray):
        l1 = np.matmul(m, self.W1) + self.b1
        if self.W2 is None:
            return l1
        return np.matmul(l1 * (l1 > 0), self.W2) + self.b2

    def context_size(self, embedding_size):
        return self.W1.shape[0]/embedding_size
