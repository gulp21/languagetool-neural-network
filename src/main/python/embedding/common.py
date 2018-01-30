import collections


def build_dataset(words, max_words_in_vocabulary: int=20000, pos_tagger=None):
    count = [['UNK', -1]]
    counter = collections.Counter(words)
    print("unique words", len(counter))
    count.extend(counter.most_common(max_words_in_vocabulary - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary
