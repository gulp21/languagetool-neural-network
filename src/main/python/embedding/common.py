import collections

from languagetool.languagetool import LanguageTool


def read_data(lt: LanguageTool, filename: str) -> [str]:
    """Read tokens from tokenized file"""
    with open(filename) as f:
        data = f.read().split()
    return data


def build_dataset(words, max_words_in_vocabulary: int=20000, pos_tagger=None):
    count = [['UNK', -1]]
    counter = collections.Counter(words)
    print("unique words", len(counter))
    most_common_counter = counter.most_common(max_words_in_vocabulary - 1)
    if pos_tagger is not None:
        vocabulary = {token for token, _ in most_common_counter}
        words_and_tags = []
        for word in words:
            if word not in vocabulary:
                words_and_tags.append(str(pos_tagger(word)))
            else:
                words_and_tags.append(word)
        tagged_counter = collections.Counter(words_and_tags).most_common()
        count.extend(tagged_counter)
    else:
        count.extend(most_common_counter)
        words_and_tags = words
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words_and_tags:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary
