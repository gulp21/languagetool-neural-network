import json

import numpy as np

base_path = "/home/markus/Dokumente/GitHub/projektarbeit/lt/languagetool-neural-network/res/cbow/eng_ordered/" #"/home/markus/Dokumente/GitHub/projektarbeit/lt/languagetool-neural-network/res/cbow/eng/"

dictionary = json.load(open(base_path + "dictionary.json"))
inverse_dictionary = {i: w for w, i in dictionary.items()}
embeddings = np.loadtxt(base_path + "embeddings.txt")
nce_weights = np.loadtxt(base_path + "nce_weights.txt")
nce_biases = np.loadtxt(base_path + "nce_biases.txt")
context_size = int(nce_weights.shape[1]/embeddings.shape[1])


def best_fits(tokenized_sentence: [str], candidates: [str]) -> [(float, str)]:
    middle_index = int(len(tokenized_sentence) / 2)
    context = tokenized_sentence[:middle_index] + tokenized_sentence[middle_index+1:]
    return best_fits_for_context(context, candidates)


def best_fits_for_context(context: [str], candidates: [str]) -> [(float, str)]:
    ps = np.reshape(embeddings[np.array(list(map(lambda w: dictionary[w], context)))], [-1]) @ nce_weights.T + nce_biases
    candidates_indices = list(map(lambda w: dictionary[w], candidates))
    order = abs(ps[candidates_indices]).argsort()
    return [((ps[candidates_indices])[i], candidates[i]) for i in order]


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
            print(context, center, p)
            score += abs(p)
        scores.append((score, candidate))
    scores.sort(key=lambda pair: pair[0])
    return scores


print(best_fits("would like to go to".split(), ["to", "too"]))
print(best_fits_with_offsetting("We would like to go to a".split(), ["to", "too"]))
print(best_fits("allow him to do business".split(), ["to", "too"]))
print(best_fits_with_offsetting("we allow him to do business with".split(), ["to", "too"]))
print(best_fits("a bit too heavy ,".split(), ["to", "too"]))
print(best_fits_with_offsetting("is a bit too heavy , so".split(), ["to", "too"]))
print(best_fits("are , too , very".split(), ["to", "too"]))
print(best_fits_with_offsetting("They are , too , very interested".split(), ["to", "too"]))
print(best_fits("I like the lecture about".split(), ["the", "then", "than"]))
print(best_fits_with_offsetting(". I like the lecture about natural".split(), ["the", "then", "than"]))
print(best_fits(", is then the processing".split(), ["the", "then", "than"]))
print(best_fits_with_offsetting("agreement , is then the processing of".split(), ["the", "then", "than"]))
print(best_fits("at the then current stock".split(), ["the", "then", "than"]))
print(best_fits_with_offsetting(", at the then current stock exchange".split(), ["the", "then", "than"]))
print(best_fits("on more than one credit".split(), ["the", "then", "than"]))
print(best_fits_with_offsetting("delay on more than one credit card".split(), ["the", "then", "than"]))
print(best_fits("nationals other than those who".split(), ["the", "then", "than"]))
print(best_fits_with_offsetting("nationals nationals other than those who do".split(), ["the", "then", "than"]))
