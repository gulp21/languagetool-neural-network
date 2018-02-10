[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

© 2017 Markus Brenneis

# For Users of LanguageTool

* Make sure you are using [LanguageTool](https://languagetool.org) version 4.0 or later.
* Download the language data from [here](https://fscs.hhu.de/languagetool/word2vec.tar.gz) and extract the archive to a directory.
* In the LanguageTool settings, choose that directory (which contains the subfolders “de”, “en” etc.) in “word2vec data directory”.
* If you are using a LanguageTool server, set `word2vecDir` in your `languagetool.cfg` (by default located at `~/.languagetool.cfg` in Linux).

# TL;DR

In case everything is already set up:

```bash
./gradlew createNGramDatabase -PlanguageCode="en-US" -PcorpusFile="training-corpus.txt" -Ptokens="to too"
python3 src/main/python/nn_words.py dictionary.txt final_embeddings.txt /tmp/to_too_training.py /tmp/to_too_validate.py .
# copy W_fc1.txt and b_fc1.txt to nuralnetwork/en/to_too
# edit nuralnetwork/en/confusion_sets.txt
# calibrate using NeuralNetworkRuleEvaluator language-code word2vec-dir RULE_ID corpus1.xml
```

# Prerequisites

## Software

This README assumes you are using an Ubuntu based operating system, but the instructions should basically also work for every other operating system.

You need Java 8 (probably alread installed if you can compile LanguageTool) and python3 with pip (the Python package manager, `sudo apt install python3-pip`). Install the following packages using pip:

* TensorFlow: machine learning library for training neural networks
* scikit-learn: machine learning library
* NumPy: scientific computing library


```
pip3 install --user tensorflow scikit-learn numpy
```

Note that TensorFlow officially supports 64 bit systems only.

If you have an nVidia GPU, you might want to use the GPU version of TensorFlow. See [tensorflow.org](https://www.tensorflow.org/install/) for installtion instructions. As the CUDA setup can take some time, I recommend proceeding with the CPU version.

## Sources

Neural network rules are supported by LanguageTool since 15 December 2017 in the development version of LanguageTool 4.0.

The code for learning new rules is not part of LanguageTool. Get it by running

```bash
git clone git@github.com:gulp21/languagetool-neural-network.git 
```

# Adding support for new languages and confusion pairs

## Getting a corpus

Whether you want to add a new language model or support for a new confusion pair, you have to get a big corpus first, which shouldn’t contain any grammar errors. Possible sources could be newspaper articles from the [Leipzig Corpora Collection](http://wortschatz.uni-leipzig.de/en/download/), [Wikipedia](http://wiki.languagetool.org/checking-the-complete-wikipedia) or [Tatoeba](https://tatoeba.org/downloads). I prefer using the Leipzig data for training and Wikipedia data for assessing rule performance. Note that newspaper and Wikipedia articles rarely include 1st and 2nd person verb forms; keep that in mind if you want to detect confusion pairs involving those verb forms.

If you just want to test your setup and don't have a corpus, yet, you can use `src/main/resources/example-corpus.txt`. Note that a good corpus should contain more than 100,000 sentences.

The training input files are plain text files containing sentences which may not be spread over multiple lines; Whether there are multiple sentences in one line doesn’t matter. If you’re using the Leipzig corpus, you can use the *-sentences.txt file, but you have to remove the line numbers first:

```bash
sed -E "s/^[0-9]+\W+//" *-sentences.txt > training-corpus.txt
```

You now have a file `training-corpus.txt` containing lots of sentences.

## Adding support for a new language

### Tokenizing the corpus

You have to tokenize the training corpus with LanguageTool. As LanguageTool itself doesn’t contain a file tokenizer for command line usage, we use a tool included in the languagetool-neural-network repository mentioned above, so `cd` to `languagetool-neural-network`. As tokenizing 1,000,000 sentences might be too much for your system memory (it can require up to 10 GB of RAM), you may decide to train the language model with fewer sentences, let’s say 300,000.

```bash
shuf training-corpus.txt | head -n300000 > language-model-corpus.txt
./gradlew tokenizeFile -PlanguageCode="en-US" -PsentencesFile="language-model-corpus.txt"
```

Don’t forget to change the `languageCode` parameter.

After having downloaded the whole internet, you should end up with a file called `language-model-corpus.txt-tokens`. The terminal output should look like this:

```
[...]
:tokenizeFile
Reading language-model-corpus.txt
Tokenizing
Tokens written to language-model-corpus.txt-tokens

BUILD SUCCESSFUL

Total time: 52.144 secs
```

### Creating a language model

If your language does not yet have a neural network rule, you have to learn a language model first, which will be shared by all neural network rules. 

First, you have to compile the word2vec C files:

```bash
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
cd src/main/python/embedding
g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd -
```

Then you can train a new language model:

```bash
python3 src/main/python/embedding/word2vec.py --train_data language-model-corpus.txt-tokens --eval_data src/main/python/embedding/question-words.txt --save_path . --epochs_to_train 10
```

If you get an error about `word2vec_ops.so`, try compiling with `-D_GLIBCXX_USE_CXX11_ABI=1` instead.

The process can take a while (on my notebook I have a rate of ~7,000 words/sec; on my university’s high performance cluster ~28,000 words/sec and a total runtime of ~30 minutes). You should see that the loss value decreases over time.

When the process has finished, you have files `dictionary.txt` (~1 MB) and `final_embeddings.txt` (~80 MB). Open the directory containing the existing word2vec models (or create a new directory, if you haven’t downloaded [models of other languages](fscs.hhu.de/languagetool/word2vec.tar.gz)), create a sub-directory `LANG` (e. g. `en`) and move the two files created to that directory.

#### What just happened?

The language model trained here is a 64 dimensional [word embedding](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/#word-embeddings). All words (or more precisely: tokens, as returned by the LanguageTool tokenizer in the previous step) which appear at least 5 times in the training corpus are mapped to a vector containing 64 numbers. Similar tokens (e. g. “I”, “you”, “he” or “my”, “your”, “her”) will magically end up “close” to each other. This will later allow the neural network to detect errors even if the exact phrase was not part of the training corpus.

The `--train_data` parameter is required, but not important for us. If you are curious, have a look at [Analogical Reasoning](https://www.tensorflow.org/tutorials/word2vec#evaluating_embeddings_analogical_reasoning).


### Adding methods to Language.java

Open the java class for your language and add the following methods:

```java
@Override
public synchronized Word2VecModel getWord2VecModel(File indexDir) throws IOException {
  return new Word2VecModel(indexDir + File.separator + getShortCode());
}

@Override
public List<Rule> getRelevantWord2VecModelRules(ResourceBundle messages, Word2VecModel word2vecModel) throws IOException {
  return NeuralNetworkRuleCreator.createRules(messages, this, word2vecModel);
}
```


## Adding support for a new confusion pair

Before you add a new confusion pair, think about whether the neural network actually has a chance to detect an error properly. The neural network gets a context of 2 tokens before and after a token as input, e. g. for the to/too pair and the sentence “I would like too learn more about neural networks.”, the network will get `[would like learn more]` as input. If you as a human can infer from `[would like learn more]` that “to” must be in the middle, the neural network can probably learn that, too. On the other hand, consider the German an/in pair and the sentence “Ich bin in der Universitätsstraße.” If you see the tokens `[Ich bin der Universitätsstraße]`, you cannot really determine whether “an” or “in” should be used, so the pair an/in is probably no good candidate for a neural network rule.

NB:

* As the neural network gets tokens as inputs, and outputs which of two tokens fits best, it cannot be used for pairs like their/they’re, because “they’re” is split into 3 tokens by LanguageTool.
* The language model is case sensitive.

### Training the neural network

First, you must generate training and validation sets from the corpus.

```bash
./gradlew createNGramDatabase -PlanguageCode="en-US" -PcorpusFile="training-corpus.txt" -Ptokens="to too"
```

Don’t forget to replace “to too” with your confusion pair tokens. The output “sampling to xxx” tells you how many training or validation samples have been created for each token.

Now you have two files, `/tmp/to_too_training.py` and `/tmp/to_too_validate.py`, and we can train a neural network:

```bash
python3 src/main/python/nn_words.py dictionary.txt final_embeddings.txt /tmp/to_too_training.py /tmp/to_too_validate.py .
```

Again, don’t forget to change the file paths to match your system.

Depending on whether a CUDA capable GPU is available and how many training samples there are, this process takes several minutes. Accuracy should get closer to 1 over time.

In the end of the training process, the neural network is validated with the validation set. The figures after “incorrect” should be near zero, and “unclassified” should not be bigger than “correct”, otherwise the learned network is probably unusable.

You now have the files `W_fc1.txt` and `b_fc1.txt` in your current working directory. Move them to `word2vec/LANG/neuralnetwork/TOKEN1_TOKEN2`, where `word2vec/LANG` is the directory containing `dictionary.txt` you have created or downloaded earlier. Don’t forget to include a `LICENSE` file if needed.

#### What just happened?

Let’s take the to/too pair as an example. You’ve trained a single-layer [neural network](https://www.quora.com/Can-you-explain-neural-nets-in-laymans-terms) which can tell you how much it feels that “to” or “too” fits into a context. E. g. given the input `[would like help you]`, it will output the “scores” `[3.95 -4.12]`, which means that it prefers the phrase “would like to help you” above “would like too help you”. On the other hand, scores like `[0.04 -0.10]` mean that the network has no preference. Which minimum score is required to mark the usage of a token as wrong is determined during the calibration of the rules, which is described in the next section. As of now, you also see those scores as part of the error message when a neural network rule detects an error.

### Adding the rule

Add a new line to `word2vec/LANG/neuralnetwork/confusion_sets.txt` which looks like this:

```
to; too; 0.5
```

If you start LanguageTool now, the rule, which has the id `LANG_to_VS_too_NEURALNETWORK`, should work, if you have specified the word2vec directory in the settings. The new rule might cause more false alarms than necessary, though.

Now you have to tweak the sensitivity of the rule, which currently is 0.5. Open `org.languagetool.dev.bigdata.NeuralNetworkRuleEvaluator` in your IDE and run the main method with the arguments `language-code word2vec-directory RULE_ID corpus1.xml corpus2.txt etc.`; a corpus can be a Wikipedia XML file or some plain text file; any number of corpora may be given. Do not use the same corpus you used for training! The output will look like this:

```
Results for both tokens
to; too; 0.50; # p=0.985, r=0.900, tp=1320, tn=1446, fp=20, fn=146, 1000+466, 2017-10-15
to; too; 0.75; # p=0.988, r=0.877, tp=1286, tn=1450, fp=16, fn=180, 1000+466, 2017-10-15
to; too; 1.00; # p=0.992, r=0.844, tp=1237, tn=1456, fp=10, fn=229, 1000+466, 2017-10-15
to; too; 1.25; # p=0.993, r=0.815, tp=1195, tn=1457, fp=9, fn=271, 1000+466, 2017-10-15
to; too; 1.50; # p=0.993, r=0.778, tp=1141, tn=1458, fp=8, fn=325, 1000+466, 2017-10-15
to; too; 1.75; # p=0.993, r=0.727, tp=1066, tn=1459, fp=7, fn=400, 1000+466, 2017-10-15
to; too; 2.00; # p=0.994, r=0.681, tp=999, tn=1460, fp=6, fn=467, 1000+466, 2017-10-15
to; too; 2.25; # p=0.996, r=0.621, tp=911, tn=1462, fp=4, fn=555, 1000+466, 2017-10-15
to; too; 2.50; # p=0.996, r=0.559, tp=820, tn=1463, fp=3, fn=646, 1000+466, 2017-10-15
to; too; 2.75; # p=0.996, r=0.489, tp=717, tn=1463, fp=3, fn=749, 1000+466, 2017-10-15
to; too; 3.00; # p=0.998, r=0.428, tp=628, tn=1465, fp=1, fn=838, 1000+466, 2017-10-15
to; too; 3.25; # p=0.998, r=0.369, tp=541, tn=1465, fp=1, fn=925, 1000+466, 2017-10-15
to; too; 3.50; # p=0.998, r=0.315, tp=462, tn=1465, fp=1, fn=1004, 1000+466, 2017-10-15
to; too; 3.75; # p=0.997, r=0.265, tp=389, tn=1465, fp=1, fn=1077, 1000+466, 2017-10-15
to; too; 4.00; # p=1.000, r=0.225, tp=330, tn=1466, fp=0, fn=1136, 1000+466, 2017-10-15

Time: 133742 ms
Recommended configuration:
to; too; 1.00                           # p=0.992, r=0.844, tp=1237, tn=1456, fp=10, fn=229, 1000+466, 2017-10-15
```

The p value is the precision, which tells you how often a detected error was an actual error (i. e. 1−p is the probability for false alarms). The r value is the recall, which tells you how often the rule could find an error. As a rule of thumb, the precision should be greater than 0.99, or 0.995 for common words/tokens. Recall should be greater than 0.5, otherwise the rule won’t detect many errors. If you have chosen a good certainty level (which is the same as the score I mentioned earlier), you can update `neuralnetwork/confusion_sets.txt`:

```
to; too; 1.00 # p=0.992, r=0.844, tp=1237, tn=1456, fp=10, fn=229, 1000+466, 2017-10-15
```

You can also pass `ALL` as rule id to evaluate the performance for all confusion sets in `confusion_sets.txt`.
