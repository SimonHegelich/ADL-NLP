# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import nltk
from nltk.corpus import gutenberg

alice = gutenberg.raw(fileids='carroll-alice.txt')
default_st = nltk.sent_tokenize
alice_sentences = default_st(text=alice)
default_wt = nltk.word_tokenize
alice_words = [default_wt(sentence.lower()) for sentence in alice_sentences]
print(len(alice_sentences))
len_alice = len(alice_sentences)

hamlet = gutenberg.raw(fileids='shakespeare-hamlet.txt')
hamlet_sentences = default_st(text=hamlet)
hamlet_words = [default_wt(sentence.lower()) for sentence in hamlet_sentences]
print(len(hamlet_sentences))
len_hamlet = len(hamlet_sentences)

sentences = alice_words + hamlet_words
len_data = len(sentences)

word2index_map = {}
index = 0
for sent in sentences:
    for word in sent:
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1

index2word_map = {index: word for word, index in word2index_map.items()}

vocabulary_size = len(index2word_map)
# Generate skip-gram pairs
skip_gram_pairs = []
win_size = 3
for sent in sentences:
    for i in range(win_size, len(sent)-win_size):
        for wn in range(win_size):
            word_context_pair = [[word2index_map[sent[i-wn]],
                                word2index_map[sent[i+wn]]],
                                word2index_map[sent[i]]]
            skip_gram_pairs.append([word_context_pair[1],
                                  word_context_pair[0][0]])
            skip_gram_pairs.append([word_context_pair[1],
                                 word_context_pair[0][1]])

batch_size = 64
embedding_dimension = 3 # Three was chosen for visualization.
negative_samples = 8
LOG_DIR = "/home/simon/Firma/AI/HegelMachine/HegelPython/logs/word2vec_intro/" # Use absolute path.

def get_skipgram_batch(batch_size):
    instance_indices = list(range(len(skip_gram_pairs)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [skip_gram_pairs[i][0] for i in batch]
    y = [[skip_gram_pairs[i][1]] for i in batch]
    return x, y

# Input data, labels
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

# Embedding lookup table currently only implemented in CPU
with tf.name_scope("embeddings"):
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_dimension],
                          -1.0, 1.0), name='embedding')
    # This is essentialy a lookup table
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Create variables for the NCE loss
nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_dimension],
                            stddev=1.0 / math.sqrt(embedding_dimension)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


loss = tf.reduce_mean(
  tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, inputs=embed, labels=train_labels,
                 num_sampled=negative_samples, num_classes=vocabulary_size))
tf.summary.scalar("NCE_loss", loss)

# Learning rate decay
global_step = tf.Variable(0, trainable=False)
learningRate = tf.train.exponential_decay(learning_rate=0.1,
                                          global_step=global_step,
                                          decay_steps=1000,
                                          decay_rate=0.95,
                                          staircase=True)
train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(LOG_DIR,
                                         graph=tf.get_default_graph())
    saver = tf.train.Saver()

    with open(os.path.join(LOG_DIR, 'metadata.tsv'), "w") as metadata:
        metadata.write('Name\tClass\n')
        for k, v in index2word_map.items():
            metadata.write('%s\t%d\n' % (v, k))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embeddings.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
    projector.visualize_embeddings(train_writer, config)

    tf.global_variables_initializer().run()

    for step in range(10000):
        x_batch, y_batch = get_skipgram_batch(batch_size)
        summary, _ = sess.run([merged, train_step],
                              feed_dict={train_inputs: x_batch,
                                         train_labels: y_batch})
        train_writer.add_summary(summary, step)

        if step % 100 == 0:
            saver.save(sess, os.path.join(LOG_DIR, "w2v_model.ckpt"), step)
            loss_value = sess.run(loss,
                                  feed_dict={train_inputs: x_batch,
                                             train_labels: y_batch})
            print("Loss at %d: %.5f" % (step, loss_value))

    # Normalize embeddings before using
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    normalized_embeddings_matrix = sess.run(normalized_embeddings)

ref_word = normalized_embeddings_matrix[word2index_map["alice"]]

cosine_dists = np.dot(normalized_embeddings_matrix, ref_word)
ff = np.argsort(cosine_dists)[::-1][1:10]
for f in ff:
    print(index2word_map[f])
    print(cosine_dists[f])