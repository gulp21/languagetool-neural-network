from embedding.common import build_dataset


def test_build_dataset():
    data, count, dictionary, reverse_dictionary = build_dataset(["a", "a", "b", "a", "b", "c", "d"], 3)
    assert count == [['UNK', 2], ('a', 3), ('b', 2)]
    assert data == [1, 1, 2, 1, 2, 0, 0]
    assert dictionary == {'b': 2, 'a': 1, 'UNK': 0}
    assert reverse_dictionary == {0: 'UNK', 2: 'b', 1: 'a'}


def test_build_dataset_with_tagger():
    tagger = lambda w: "TAG"
    data, count, dictionary, reverse_dictionary = build_dataset(["a", "a", "b", "a", "a", "b", "b", "c", "d"], 3, tagger)
    assert count == [['UNK', 0], ('a', 4), ('b', 3), ('TAG', 2)]
    assert data == [1, 1, 2, 1, 1, 2, 2, 3, 3]
    assert dictionary == {'b': 2, 'a': 1, 'UNK': 0, 'TAG': 3}
    assert reverse_dictionary == {0: 'UNK', 2: 'b', 1: 'a', 3: 'TAG'}
