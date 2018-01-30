# References
# - https://www.tensorflow.org/versions/r0.10/tutorials/word2vec/index.html
# - https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math

import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin

# Step 1: Download the data.
from embedding.common import build_dataset

filename = "/tmp/c-tokens"


# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with open(filename) as f:
        data = tf.compat.as_str(f.read()).split()
    return data


words = read_data(filename)
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.

data, count, dictionary, reverse_dictionary = build_dataset(words)
vocabulary_size = len(dictionary)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, context_window):
    # all context tokens should be used, hence no associated num_skips argument
    global data_index
    context_size = 2 * context_window
    batch = np.ndarray(shape=(batch_size, context_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * context_window + 1  # [ context_window target context_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size):
        # context tokens are just all the tokens in buffer except the target
        batch[i, :] = [token for idx, token in enumerate(buffer) if idx != context_window]
        labels[i, 0] = buffer[context_window]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=8, context_window=1)
for i in range(8):
    print(batch[i, 0], reverse_dictionary[batch[i, 0]],
          batch[i, 1], reverse_dictionary[batch[i, 1]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
context_window = 2  # How many words to consider left and right.
context_size = 2 * context_window

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
# valid_target_examples = [[dictionary["d"], dictionary["like"], dictionary["go"], dictionary["to"]],
#                          [dictionary[","], dictionary["and"], dictionary["we"], dictionary["will"]]]
valid_target_examples = [[dictionary["wir"], dictionary["sind"], dictionary["Großen"], dictionary["und"]],
                         [dictionary["es"], dictionary["ist"], dictionary["zu"], dictionary["sehen"]]]
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size, context_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    valid_target_dataset = tf.constant(valid_target_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        # take mean of embeddings of context words for context embedding
        embed_context = tf.reshape(embed, [batch_size, embedding_size * context_size])

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size * context_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed_context, num_sampled, vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    valid_target_probabilities = tf.matmul(tf.reshape(tf.nn.embedding_lookup(embeddings, valid_target_dataset), [len(valid_target_examples), embedding_size * context_size]), nce_weights, transpose_b=True) + nce_biases

    # Add variable initializer.
    init = tf.initialize_all_variables()

# Step 5: Begin training.
num_steps = 200001

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, context_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)

            pss = valid_target_probabilities.eval()
            for context, ps in zip(valid_target_examples, pss):
                top_k = 8
                nearest = (-ps).argsort()[1:top_k + 1]
                context_words = list(map(lambda w: reverse_dictionary[w], context))
                nearest_words = list(map(lambda w: reverse_dictionary[w], nearest))
                print("Nearest to %s: %s" % (str(context_words), str(nearest_words)))


    final_embeddings = normalized_embeddings.eval()

    json.dump(dictionary, open("/tmp/dictionary.json", "w"))
    np.savetxt("/tmp/embeddings.txt", embeddings.eval())
    np.savetxt("/tmp/nce_weights.txt", nce_weights.eval())
    np.savetxt("/tmp/nce_biases.txt", nce_biases.eval())

# Step 6: Visualize the embeddings.

# def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
#     assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
#     plt.figure(figsize=(18, 18))  # in inches
#     for i, label in enumerate(labels):
#         x, y = low_dim_embs[i, :]
#         plt.scatter(x, y)
#         plt.annotate(label,
#                      xy=(x, y),
#                      xytext=(5, 2),
#                      textcoords='offset points',
#                      ha='right',
#                      va='bottom')
#
#     plt.savefig(filename)
#
#
# try:
#     from sklearn.manifold import TSNE
#     import matplotlib.pyplot as plt
#
#     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#     plot_only = 500
#     low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
#     labels = [reverse_dictionary[i] for i in xrange(plot_only)]
#     plot_with_labels(low_dim_embs, labels)
#
# except ImportError:
#     print("Please install sklearn and matplotlib to visualize embeddings.")