import json

import numpy as np

# base_path = "/home/markus/Dokumente/GitHub/projektarbeit/lt/languagetool-neural-network/res/cbow/eng_ordered/" #"/home/markus/Dokumente/GitHub/projektarbeit/lt/languagetool-neural-network/res/cbow/eng/"
base_path = "/home/markus/Dokumente/GitHub/projektarbeit/lt/languagetool-neural-network/res/cbow/deu_ordered/" #"/home/markus/Dokumente/GitHub/projektarbeit/lt/languagetool-neural-network/res/cbow/eng/"

dictionary = json.load(open(base_path + "dictionary.json"))
inverse_dictionary = {i: w for w, i in dictionary.items()}
embeddings = np.loadtxt(base_path + "embeddings.txt")
nce_weights = np.loadtxt(base_path + "nce_weights.txt")
nce_biases = np.loadtxt(base_path + "nce_biases.txt")
context_size = int(nce_weights.shape[1]/embeddings.shape[1])
context_window = int(context_size / 2)


def best_fits(tokenized_sentence: [str], candidates: [str]) -> [(float, str)]:
    middle_index = int(len(tokenized_sentence) / 2)
    context = tokenized_sentence[:middle_index] + tokenized_sentence[middle_index+1:]
    return best_fits_for_context(context, candidates)


def best_fits_for_context(context: [str], candidates: [str]) -> [(float, str)]:
    ps = np.reshape(embeddings[np.array(get_embedding_indices(context))], [-1]) @ nce_weights.T + nce_biases
    candidates_indices = get_embedding_indices(candidates)
    order = abs(ps[candidates_indices]).argsort()
    return [((ps[candidates_indices])[i], candidates[i]) for i in order]


def get_embedding_indices(candidates):
    return list(map(lambda w: dictionary[w] if w in dictionary else dictionary["UNK"], candidates))


def best_fits_with_offsetting(tokenized_sentence: [str], candidates: [str]) -> [(float, str)]:
    middle_index = int(len(tokenized_sentence) / 2)
    scores = []
    for candidate in candidates:
        tokenized_candidate_sentence = tokenized_sentence[:middle_index] + [candidate] + tokenized_sentence[middle_index+1:]
        score = 1
        for offset in range(len(tokenized_sentence) - context_size):
            context = tokenized_candidate_sentence[offset:int(context_size/2)+offset] + tokenized_candidate_sentence[int(context_size/2)+1+offset:int(context_size/2)+1+offset+int(context_size/2)]
            center = tokenized_candidate_sentence[int(context_size/2)+offset]
            [(p, _)] = best_fits_for_context(context, [center])
            # print(context, center, p)
            score += abs(p)
        scores.append((score, candidate))
    scores.sort(key=lambda pair: pair[0])
    return scores


def print_measures(error_detection_tp, error_detection_fp, error_detection_tn, error_detection_fn):
    accs = 0
    ps = 0
    rs = 0
    for key in error_detection_tp.keys():
        tp = error_detection_tp[key]
        fp = error_detection_fp[key]
        tn = error_detection_tn[key]
        fn = error_detection_fn[key]
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        print(key, (tp + fn), "acc", accuracy, "p", precision, "r", recall)
        accs += accuracy
        ps += precision
        rs += recall
    n_keys = len(error_detection_tp.keys())
    print("all", "acc", accs/n_keys, "ps", ps/n_keys, "rs", rs/n_keys)


def evaluate_text(tokenized_sentences: [str], candidates: [str], threshold: float=1) -> float:
    tp = 0
    fp = 0
    error_detection_tp = {token: 0 for token in candidates}
    error_detection_fp = {token: 0 for token in candidates}
    error_detection_tn = {token: 0 for token in candidates}
    error_detection_fn = {token: 0 for token in candidates}
    for i, token in enumerate(tokenized_sentences[context_window+1:-context_window-1], context_window+1):
        if token in candidates:
            sentence = tokenized_sentences[i - context_window : i + context_window + 1]
            best_fits = best_fits_with_offsetting(sentence, candidates)
            (score_best, best_fit) = best_fits[0]
            (score_second, second_fit) = best_fits[1]
            if best_fit == token:
                tp += 1
                error_detection_tn[second_fit] += 1
                if score_second - score_best > threshold:
                    error_detection_tp[best_fit] += 1
            else:
                fp += 1
                error_detection_fn[second_fit] += 1
                if score_second - score_best > threshold:
                    error_detection_fp[best_fit] += 1
                print("false positive", best_fit, sentence)
                print_measures(error_detection_tp, error_detection_fp, error_detection_tn, error_detection_fn)
    return tp / (tp + fp)


# print(best_fits("would like to go to".split(), ["to", "too"]))
# print(best_fits_with_offsetting("We would like to go to a".split(), ["to", "too"]))
# print(best_fits("allow him to do business".split(), ["to", "too"]))
# print(best_fits_with_offsetting("we allow him to do business with".split(), ["to", "too"]))
# print(best_fits("a bit too heavy ,".split(), ["to", "too"]))
# print(best_fits_with_offsetting("is a bit too heavy , so".split(), ["to", "too"]))
# print(best_fits("are , too , very".split(), ["to", "too"]))
# print(best_fits_with_offsetting("They are , too , very interested".split(), ["to", "too"]))
# print(best_fits("I like the lecture about".split(), ["the", "then", "than"]))
# print(best_fits_with_offsetting(". I like the lecture about natural".split(), ["the", "then", "than"]))
# print(best_fits(", is then the processing".split(), ["the", "then", "than"]))
# print(best_fits_with_offsetting("agreement , is then the processing of".split(), ["the", "then", "than"]))
# print(best_fits("at the then current stock".split(), ["the", "then", "than"]))
# print(best_fits_with_offsetting(", at the then current stock exchange".split(), ["the", "then", "than"]))
# print(best_fits("on more than one credit".split(), ["the", "then", "than"]))
# print(best_fits_with_offsetting("delay on more than one credit card".split(), ["the", "then", "than"]))
# print(best_fits("nationals other than those who".split(), ["the", "then", "than"]))
# print(best_fits_with_offsetting("nationals nationals other than those who do".split(), ["the", "then", "than"]))

print(best_fits_with_offsetting(", wenn sie schon alle Zusammenhänge etwas".split(), ["schon", "schön"]))
print(best_fits_with_offsetting("das eigentlich auch schon , wir haben".split(), ["schon", "schön"]))
print(best_fits_with_offsetting("zu selten , schön und wichtig um".split(), ["schon", "schön"]))
print(best_fits_with_offsetting(", die nicht schön blühen , noch".split(), ["schon", "schön"]))
