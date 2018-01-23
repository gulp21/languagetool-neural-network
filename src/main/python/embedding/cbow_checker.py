import json

import numpy as np

base_path = "/home/markus/Dokumente/GitHub/projektarbeit/lt/languagetool-neural-network/res/cbow/eng/"

dictionary = json.load(open(base_path + "dictionary.json"))
inverse_dictionary = {i: w for w, i in dictionary.items()}
embeddings = np.loadtxt(base_path + "embeddings.txt")
nce_weights = np.loadtxt(base_path + "nce_weights.txt")
nce_biases = np.loadtxt(base_path + "nce_biases.txt")

def best_fits(tokenized_sentence: [str], candidates: [str]) -> [(float, str)]:
    middle_index = int(len(tokenized_sentence) / 2)
    context = tokenized_sentence[:middle_index] + tokenized_sentence[middle_index+1:]
    ps = np.sum(embeddings[np.array(list(map(lambda w: dictionary[w], context)))], 0) @ nce_weights.T + nce_biases
    candidates_indices = list(map(lambda w: dictionary[w], candidates))
    order = abs(ps[candidates_indices]).argsort()
    return [((ps[candidates_indices])[i], candidates[i]) for i in order]

print(best_fits("would like to go to".split(), ["to", "too"]))
print(best_fits("allow him to do business".split(), ["to", "too"]))
print(best_fits("a bit too heavy ,".split(), ["to", "too"]))
print(best_fits("are , , very".split(), ["to", "too"]))
print(best_fits("I like the lecture about".split(), ["the", "then", "than"]))
print(best_fits(", is then the processing".split(), ["the", "then", "than"]))
print(best_fits("at the then current stock".split(), ["the", "then", "than"]))
print(best_fits("on more than one credit".split(), ["the", "then", "than"]))
print(best_fits("( other than those who".split(), ["the", "then", "than"]))
