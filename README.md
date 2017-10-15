[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

© 2017 Markus Brenneis

# Prerequisites

This README assumes you are using an Ubuntu based operating system, but the instructions should basically also work for every other Unix-like operating system. If you’re using Windows: Sorry, I can’t give you detailed instructions.

TODO
java8
python3

# Adding support for new languages and confusion pairs

## Getting a corpus

Whether you want to add a new language model or support for a new confusion pair, you have to get a big corpus first, which shouldn’t contain any grammar errors. Possible sources could be newspaper articles from the Leipzig Corpora Collection http://wortschatz.uni-leipzig.de/en/download/ or Wikipedia http://wiki.languagetool.org/checking-the-complete-wikipedia. I prefer using the Leipzig data for training and Wikipedia data for assessing rule performance.

The training input files are plain text files containing sentences which may not be spread over multiple lines. If you’re using the Leipzig corpus, you can use the *-sentences.txt file, but you have to remove the line numbers first:

```bash
sed -E "s/^[0-9]+\W+//" *-sentences.txt > training-corpus.txt
```

You now have a file `training-corpus.txt` containing lots of sentences. Now we have to tokenize the file with LanguageTool.

If you have not already done it, clone the languagetool-neural-network (`git clone TODO`) and `cd` to `languagetool-neural-network`. As tokenizing 1,000,000 sentences might be too much for your memory, you may decide to train the language model with fewer sentences, let’s say 300,000.

```bash
shuf training-corpus.txt | head -n300000 > language-model-corpus.txt
./gradlew tokenizeFile -PlanguageCode="en-US" -PsentencesFile="language-model-corpus.txt"
```

After downloading the whole internet, you should end up with a file called `language-model-corpus.txt-tokens`. The terminal output should look like this:

```
[...]
:tokenizeFile
Reading language-model-corpus.txt
Tokenizing
Tokens written to language-model-corpus.txt-tokens

BUILD SUCCESSFUL

Total time: 52.144 secs
```

## Creating a language model

If your language does not yet have a neural network rule, you have to learn a language model first, which will be shared by all neural network rules. 

First, you have to compile the word2vec c files:

```bash
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
cd src/main/python/embedding
g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=1
cd -
```

Then you can train a new language model:

```bash
python3 src/main/python/embedding/word2vec.py --train_data language-model-corpus.txt-tokens --eval_data src/main/python/embedding/question-words.txt --save_path . --epochs_to_train 1
```

If you get something like `tensorflow.python.framework.errors_impl.NotFoundError: word2vec_ops.so: undefined symbol:_ZN10tensorflow7strings6StrCatB5cxx11ERKNS0_8AlphaNumE`, go back to the `g++` step and use `-D_GLIBCXX_USE_CXX11_ABI=0`.

The process can take a while. You should see that the loss value decreases over time.


### What just happened?

The language model trained here is a 64 dimensional [word embedding](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/#word-embeddings). All words which appear at least 5 times in the training corpus are mapped to a vector containing 64 numbers. Similar words (e. g. “I”, “you”, “he” or “my”, “your”, “her”) will magically end up “close” to each other. This will later allow the neural network to detect errors even if the exact phrase was not part of the training corpus.

The `--train_data` parameter is required, but not important for us. If you are curious, have a look at [Analogical Reasoning](https://www.tensorflow.org/tutorials/word2vec#evaluating_embeddings_analogical_reasoning).

## Adding support for a new confusion pair

Before you add a new confusion pair, think about whether the neural network actually has a chance to detect an error properly. The neural network gets a context of 2 words before and after a token as input, e. g. for the to/too pair and the sentence “I would like too learn more about neural networks.”, the network will get `[would like learn more]` as input. If you as a human can infer from `[would like learn more]` that “to” must be in the middle, the neural network can probably learn that, too. On the other hand, consider the German an/in pair and the sentence “Ich bin in der Universitätsstraße.” If you see the tokens `[Ich bin der Universitätsstraße]`, you cannot really determine whether “an” or “in” should be used, so the pair an/in is probably no good candidate for a neural network rule.

NB:

* As the neural network gets tokens as inputs and outputs which of two tokens is fits best, it cannot be used for pairs like their/they’re, because “they’re” are 3 tokens.
* The language model is case sensitive.
* As the networks gets two word before and after a token as input, detcting errors at the very beginning or end of a sentence is currently not possible. (This can later be mitigated by introducing special BEFORE_SENTENCE and AFTER_SENTENCE tokens).
