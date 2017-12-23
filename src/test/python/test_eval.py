from eval import get_relevant_ngrams, evaluate_ngrams, similar_words


def test_get_single_relevant_ngram():
    ngrams = get_relevant_ngrams("a b c d e", ["c"])
    assert ngrams == [["a", "b", "c", "d", "e"]]


def test_get_two_relevant_ngrams():
    ngrams = get_relevant_ngrams("a b c d e f", ["c", "d"])
    assert ngrams == [["a", "b", "c", "d", "e"], ["b", "c", "d", "e", "f"]]


def test_evaluate_one_ngram():
    eval_result = evaluate_ngrams([["a", "b", "c", "d", "e"]], ["c", "d"], lambda _: True)
    assert eval_result.tp == 1
    assert eval_result.fp == 1
    assert eval_result.tn == 0
    assert eval_result.fn == 0


def test_evaluate_ngrams():
    eval_result = evaluate_ngrams([["a", "b", "c", "d", "e"], ["b", "c", "d", "e", "f"]], ["c", "d", "f"],
                                  lambda ng: not(ng[0] == "a" and ng[2] == "c"))
    assert eval_result.tp == 4
    assert eval_result.fp == 1
    assert eval_result.tn == 1
    assert eval_result.fn == 0


def test_similar_words():
    selected_words = similar_words("dein", ["sein", "dumme", "dein", "deinen", "dienend"])
    assert selected_words == ["sein", "dein", "deinen"]


def test_similar_short_words():
    selected_words = similar_words("da", ["da", "dann", "der", "aus"])
    assert selected_words == ["da", "dann", "der"]
