from embedding.common import build_dataset


def test_build_dataset():
    data, count, dictionary, reverse_dictionary = build_dataset(["a", "a", "b", "a", "b", "c", "d"], 3)
    assert count == [['UNK', 2], ('a', 3), ('b', 2)]
    assert data == [1, 1, 2, 1, 2, 0, 0]
    assert dictionary == {'b': 2, 'a': 1, 'UNK': 0}
    assert reverse_dictionary == {0: 'UNK', 2: 'b', 1: 'a'}
