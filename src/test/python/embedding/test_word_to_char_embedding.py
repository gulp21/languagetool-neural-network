import numpy as np
from numpy.testing import assert_array_almost_equal

from embedding.word_to_char_embedding import to_char_embedding


def test_to_char_embedding():
    dictionary = {"ab": 0, "bc": 1}
    embedding = np.asarray([[1, 2, 3], [3, 4, 5]])
    char_dictionary, char_embedding = to_char_embedding(dictionary, embedding)
    assert_array_almost_equal(char_embedding[char_dictionary["UNK"]], [0, 0, 0])
    assert_array_almost_equal(char_embedding[char_dictionary["a"]], [1, 2, 3])
    assert_array_almost_equal(char_embedding[char_dictionary["b"]], [2, 3, 4])
    assert_array_almost_equal(char_embedding[char_dictionary["c"]], [3, 4, 5])
