[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

© 2017 Markus Brenneis

# TL;DR

In case everything is already set up:

```bash
./create_training_files.bash training-corpus.txt to too
./create_classifier.bash dictionary.txt final_embeddings.txt /tmp/to_too_training.py /tmp/to_too_validate.py .
# copy W_fc1.txt and b_fc1.txt to resources
# edit nuralnetwork/confusion_sets.txt
# calibrate using NeuralNetworkRuleEvaluator language-code word2vec-dir RULE_ID corpus1.xml
```

# Prerequisites

## Software

This README assumes you are using an Ubuntu based operating system, but the instructions should basically also work for every other Unix-like operating system. If you’re using Windows: Sorry, I can’t give you detailed instructions.

You need Java 8 (probably alread installed if you can compile LanguageTool) and python3 with pip (the Python package manager, `sudo apt install python3-pip`). Install the following packages using pip:

* TensorFlow: machine learning library for training neural networks
* scikit-learn: machine learning library
* NumPy: scientific computing library


```
pip3 install --user tensorflow scikit-learn numpy
```

If you have an nVidia GPU, you might want to use the GPU version of TensorFlow. See [tensorflow.org](https://www.tensorflow.org/install/) for installtion instructions. As the CUDA setup can take some time, I recommend proceeding with the CPU version.

## Sources

Neural network rules are not yet official part of LanguageTool, but currently developed on a branch in @gulp21's repository. `cd` into your LanguageTool source directory and do

```bash
git remote add gulp21 git@github.com:gulp21/languagetool.git
git fetch gulp21
git checkout neuralnetworkrule
```

The code for learning new rules is not part of LanguageTool. Get it by running

```bash
git clone git@github.com:gulp21/languagetool-neural-network.git 
```

# Adding support for new languages and confusion pairs

## Getting a corpus

Whether you want to add a new language model or support for a new confusion pair, you have to get a big corpus first, which shouldn’t contain any grammar errors. Possible sources could be newspaper articles from the [Leipzig Corpora Collection](http://wortschatz.uni-leipzig.de/en/download/) or [Wikipedia](http://wiki.languagetool.org/checking-the-complete-wikipedia). I prefer using the Leipzig data for training and Wikipedia data for assessing rule performance. Note that newspaper and Wikipedia articles rarely include 1st and 2nd person verb forms; keep that in mind if you want to detect confusion pairs involving those verb forms.

The training input files are plain text files containing sentences which may not be spread over multiple lines. If you’re using the Leipzig corpus, you can use the *-sentences.txt file, but you have to remove the line numbers first:

```bash
sed -E "s/^[0-9]+\W+//" *-sentences.txt > training-corpus.txt
```

You now have a file `training-corpus.txt` containing lots of sentences.

## Adding support for a new language

### Tokenizing the corpus

You have to tokenize the training corpus with LanguageTool. `cd` to `languagetool-neural-network`. As tokenizing 1,000,000 sentences might be too much for your memory, you may decide to train the language model with fewer sentences, let’s say 300,000.

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
python3 src/main/python/embedding/word2vec.py --train_data language-model-corpus.txt-tokens --eval_data src/main/python/embedding/question-words.txt --save_path . --epochs_to_train 1
```

If you get an error about `word2vec_ops.so`, try compiling with `-D_GLIBCXX_USE_CXX11_ABI=1` instead.

The process can take a while (on my notebook I have a rate of ~7,000 words/sec; on my university’s high performance cluster ~28,000 words/sec and a total runtime of ~30 minutes). You should see that the loss value decreases over time.

When the process has finished, you have files `dictionary.txt` (~1 MB) and `final_embeddings.txt` (~80 MB). Open the directory containing the existing word2vec models (or create a new directory, if you haven’t download models of other languages), create a sub-directory `LANG` (e. g. `en`) and move the two created to that directory.

#### What just happened?

The language model trained here is a 64 dimensional [word embedding](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/#word-embeddings). All words which appear at least 5 times in the training corpus are mapped to a vector containing 64 numbers. Similar words (e. g. “I”, “you”, “he” or “my”, “your”, “her”) will magically end up “close” to each other. This will later allow the neural network to detect errors even if the exact phrase was not part of the training corpus.

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

Before you add a new confusion pair, think about whether the neural network actually has a chance to detect an error properly. The neural network gets a context of 2 words before and after a token as input, e. g. for the to/too pair and the sentence “I would like too learn more about neural networks.”, the network will get `[would like learn more]` as input. If you as a human can infer from `[would like learn more]` that “to” must be in the middle, the neural network can probably learn that, too. On the other hand, consider the German an/in pair and the sentence “Ich bin in der Universitätsstraße.” If you see the tokens `[Ich bin der Universitätsstraße]`, you cannot really determine whether “an” or “in” should be used, so the pair an/in is probably no good candidate for a neural network rule.

NB:

* As the neural network gets tokens as inputs, and outputs which of two tokens is fits best, it cannot be used for pairs like their/they’re, because “they’re” are 3 tokens.
* The language model is case sensitive.
* As the networks gets two words before and after a token as input, detecting errors at the very beginning or end of a sentence is currently not possible. (This can later be mitigated by introducing special BEFORE_SENTENCE and AFTER_SENTENCE tokens).

### Training the neural network

First, you must generate training and validation sets from the corpus.

```bash
./create_training_files.bash training-corpus.txt to too
```

Don’t forget to replace “to” and “too” with your confusion set tokens. The output “undersampling to xxx” tells you how many training or validation samples have been created for each token.

Now you have two files, `/tmp/to_too_training.py` and `/tmp/to_too_validate.py`, and we can train a neural network:

```bash
./create_classifier.bash dictionary.txt final_embeddings.txt /tmp/to_too_training.py /tmp/to_too_validate.py .
```

Depending on whether a CUDA capable GPU is available and how many training samples there are, this process takes several minutes. Accuracy should get closer to 1 over time.

In the end of the training process, the neural network is validated with the validation set. The figures after “incorrect” should be near zero, and “unclassified” should not be bigger than “correct”, otherwise the learned network is probably unusable. (NB: The validation does not use the same algorithm as the `NeuralNetworkRule` in LanguageTool, but it still is a good hint whether the network learned something sensible or not.)

You now have the files `W_fc1.txt` and `b_fc1.txt` in your current working directory. Move them to `languagetool-language-modules/LANG/src/main/resources/org/languagetool/resource/LANG/neuralnetwork/TOKEN1_TOKEN2`. Don’t forget to include a `LICENSE` file if needed.

#### What just happened?

Let’s take the to/too pair as an example. You’ve trained a single-layer [neural network](https://www.quora.com/Can-you-explain-neural-nets-in-laymans-terms) which can tell you how much it feels that “to” or “too” fits into a context. E. g. given the input `[would like help you]`, it will output the “scores” `[3.95 -4.12]`, which means that it prefers the phrase “would like to help you” above “would like too help you”. On the other hand, scores like `[0.04 -0.10]` mean that the network has no preference. Which minimum score is required to mark the usage of a token as wrong is determined during the calibration of the rules, which is described in the next section. As of now, you also see those scores as part of the error message when a neural network rule detects an error.

### Adding the rule

Add a new line to `languagetool-language-modules/LANG/src/main/resources/org/languagetool/resource/LANG/neuralnetwork/confusion_sets.txt` which looks like this:

```
to; too; 0.5
```

If you build LanguageTool now, the rule, which has the id `LANG_to_VS_too_NEURALNETWORK`, should work, if you have specified the word2vec directory in the settings. The new rule might cause more false alarms than necessary, though.

```bash
./build.sh languagetool-standalone package -DskipTests
java -jar languagetool-standalone/target/LanguageTool-3.9-SNAPSHOT/LanguageTool-3.9-SNAPSHOT/languagetool.jar
```

Now you have to tweak the sensitivity of the rule, which currently is 0.5. Open `org.languagetool.dev.bigdata.NeuralNetworkRuleEvaluator` in your IDE and run the main method with the arguments `language-code word2vec-directory RULE_ID corpus1.xml corpus2.txt etc.`; a corpus can be a Wikipedia XML file or some plain text file and any number of corpora may be given. Do not use the same corpus you used for training! The output will look like this:

```
Evaluation results for to/too with 1466 sentences as of Sun Oct 15 21:15:16 CEST 2017:

Certainty: 0.50 - 20 false positives, 146 false negatives, 1320 true positives, 1446 true negatives
to; too; 0.50; # p=0.985, r=0.900, tp=1320, tn=1446, fp=20, fn=146, 1000+466, 2017-10-15

Certainty: 0.75 - 16 false positives, 180 false negatives, 1286 true positives, 1450 true negatives
to; too; 0.75; # p=0.988, r=0.877, tp=1286, tn=1450, fp=16, fn=180, 1000+466, 2017-10-15

Certainty: 1.00 - 10 false positives, 229 false negatives, 1237 true positives, 1456 true negatives
to; too; 1.00; # p=0.992, r=0.844, tp=1237, tn=1456, fp=10, fn=229, 1000+466, 2017-10-15

Certainty: 1.25 - 9 false positives, 271 false negatives, 1195 true positives, 1457 true negatives
to; too; 1.25; # p=0.993, r=0.815, tp=1195, tn=1457, fp=9, fn=271, 1000+466, 2017-10-15

Certainty: 1.50 - 8 false positives, 325 false negatives, 1141 true positives, 1458 true negatives
to; too; 1.50; # p=0.993, r=0.778, tp=1141, tn=1458, fp=8, fn=325, 1000+466, 2017-10-15

Certainty: 1.75 - 7 false positives, 400 false negatives, 1066 true positives, 1459 true negatives
to; too; 1.75; # p=0.993, r=0.727, tp=1066, tn=1459, fp=7, fn=400, 1000+466, 2017-10-15

Certainty: 2.00 - 6 false positives, 467 false negatives, 999 true positives, 1460 true negatives
to; too; 2.00; # p=0.994, r=0.681, tp=999, tn=1460, fp=6, fn=467, 1000+466, 2017-10-15

Certainty: 2.25 - 4 false positives, 555 false negatives, 911 true positives, 1462 true negatives
to; too; 2.25; # p=0.996, r=0.621, tp=911, tn=1462, fp=4, fn=555, 1000+466, 2017-10-15

Certainty: 2.50 - 3 false positives, 646 false negatives, 820 true positives, 1463 true negatives
to; too; 2.50; # p=0.996, r=0.559, tp=820, tn=1463, fp=3, fn=646, 1000+466, 2017-10-15

Certainty: 2.75 - 3 false positives, 749 false negatives, 717 true positives, 1463 true negatives
to; too; 2.75; # p=0.996, r=0.489, tp=717, tn=1463, fp=3, fn=749, 1000+466, 2017-10-15

Certainty: 3.00 - 1 false positives, 838 false negatives, 628 true positives, 1465 true negatives
to; too; 3.00; # p=0.998, r=0.428, tp=628, tn=1465, fp=1, fn=838, 1000+466, 2017-10-15

Certainty: 3.25 - 1 false positives, 925 false negatives, 541 true positives, 1465 true negatives
to; too; 3.25; # p=0.998, r=0.369, tp=541, tn=1465, fp=1, fn=925, 1000+466, 2017-10-15

Certainty: 3.50 - 1 false positives, 1004 false negatives, 462 true positives, 1465 true negatives
to; too; 3.50; # p=0.998, r=0.315, tp=462, tn=1465, fp=1, fn=1004, 1000+466, 2017-10-15

Certainty: 3.75 - 1 false positives, 1077 false negatives, 389 true positives, 1465 true negatives
to; too; 3.75; # p=0.997, r=0.265, tp=389, tn=1465, fp=1, fn=1077, 1000+466, 2017-10-15

Certainty: 4.00 - 0 false positives, 1136 false negatives, 330 true positives, 1466 true negatives
to; too; 4.00; # p=1.000, r=0.225, tp=330, tn=1466, fp=0, fn=1136, 1000+466, 2017-10-15
```

The p value is the precision, which tells you how often a detected error was an actual error (i. e. 1−p is the probability for false alarms). The r value is the recall, which tells you how often the rule could find an error. As a rule of thumb, the precision should be greater than 0.99, or 0.995 for common words. Recall should be greater than 0.5, otherwise the rule won’t detect many errors. If you have chosen a good certainty level (which is the same as the score I mentioned earlier), you can update `neuralnetwork/pconfusion_sets.txt`:

```
to; too; 1.00 # p=0.992, r=0.844, tp=1237, tn=1456, fp=10, fn=229, 1000+466, 2017-10-15
```
