import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import gutenberg

default_st = nltk.sent_tokenize
default_wt = nltk.word_tokenize

alice = gutenberg.raw(fileids='carroll-alice.txt')
alice_sentences = default_st(text=alice)
alice_tuples = [nltk.pos_tag(default_wt(sentence)) for sentence in alice_sentences]
alice_words = [[word[0] for word in sentence] for sentence in alice_tuples]
alice_tags = [[word[1] for word in sentence] for sentence in alice_tuples]
alice_len = len(alice_sentences)

melville = gutenberg.raw(fileids='melville-moby_dick.txt')
melville_sentences = default_st(melville)
melville_tuples = [nltk.pos_tag(default_wt(sentence)) for sentence in melville_sentences]
melville_words = [[word[0] for word in sentence] for sentence in melville_tuples]
melville_tags = [[word[1] for word in sentence] for sentence in melville_tuples]
melville_len = len(melville_sentences)

austin = gutenberg.raw(fileids='austen-sense.txt')
austin_sentences = default_st(austin)
austin_tuples = [nltk.pos_tag(default_wt(sentence)) for sentence in austin_sentences]
austin_words = [[word[0] for word in sentence] for sentence in austin_tuples]
austin_tags = [[word[1] for word in sentence] for sentence in austin_tuples]
austin_len = len(austin_sentences)

data = np.array(alice_words + melville_words + austin_words)
data_tags = np.array(alice_tags + melville_tags + austin_tags)
data_len = len(data)

max_len = 0
for sent in data:
    if len(sent)>max_len:
        max_len = len(sent)

batch_size = 100
embedding_dimension = 64
embedding_dimension_tags = 32
num_classes = 3
hidden_layer_size = 128
num_LSTM_layers = 4
hidden_layer_size_tags = 64
num_LSTM_layers_tags = 2
#element_size = 1
epochs = 1500

seqlens = []

for sentence_id in range(data_len):
    seqlens.append(len(data[sentence_id]))

    if len(data[sentence_id]) < max_len:
        pads = ['PAD']*(max_len-len(data[sentence_id]))
        data[sentence_id] = data[sentence_id] + pads

### tags ###
for sentence_id in range(data_len):
    if len(data_tags[sentence_id]) < max_len:
        pads = ['PAD']*(max_len-len(data_tags[sentence_id]))
        data_tags[sentence_id] = data_tags[sentence_id] + pads

####

labels = [2] * alice_len + [1] * melville_len + [0] * austin_len
# labels = [2] * 100 + [1] * 100 + [0] * 100
# labels_hot = tf.one_hot(labels, depth=num_classes)

for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0]*num_classes
    one_hot_encoding[label] = 1
    labels[i] = one_hot_encoding

word2index_map = {}
index = 0
for sent in data:
    for word in sent:
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1

index2word_map = {index: word for word, index in word2index_map.items()}

vocabulary_size = len(index2word_map)

#### tags ####

word2index_map_tags = {}
index = 0
for sent in data_tags:
    for tag in sent:
        if tag not in word2index_map_tags:
            word2index_map_tags[tag] = index
            index += 1

index2word_map_tags = {index: tag for tag, index in word2index_map_tags.items()}

vocabulary_size_tags = len(index2word_map_tags)

###

train_size = int(data_len/2) # has to be integer for slicing array
data_indices = list(range(len(data)))
np.random.shuffle(data_indices)
data = np.array(data)[data_indices]
labels = np.array(labels)[data_indices]
seqlens = np.array(seqlens)[data_indices]
train_x = data[:train_size] # added dimension of array
train_y = labels[:train_size]
train_seqlens = seqlens[:train_size]

test_x = data[train_size:]
test_y = labels[train_size:]
test_seqlens = seqlens[train_size:]

#### tags ###
data_tags = np.array(data_tags)[data_indices]
train_x_tags = data_tags[:train_size]
test_x_tags = data_tags[train_size:]
test_y = labels[train_size:]
test_seqlens = seqlens[train_size:]

def get_sentence_batch(batch_size, data_x,
                       data_y, data_seqlens, data_x_tags):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [[word2index_map[word] for word in data_x[i]]
         for i in batch]
    x2 = [[word2index_map_tags[word] for word in data_x_tags[i]]
         for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x, y, seqlens, x2


_inputs = tf.placeholder(tf.int32, shape=[batch_size, max_len], name='Input')
_labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes], name='Labels')
# seqlens for dynamic calculation
_seqlens = tf.placeholder(tf.int32, shape=[batch_size], name='Seqlens')
_inputs_tags = tf.placeholder(tf.int32, shape=[batch_size, max_len], name='Input_tags')

with tf.name_scope("embeddings"):
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size,
                           embedding_dimension],
                          -1.0, 1.0), name='embedding')
    embed = tf.nn.embedding_lookup(embeddings, _inputs)

with tf.variable_scope("lstm"):
    # Define a function that gives the output in the right shape
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, forget_bias=1.0)
    cell = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell() for _ in range(num_LSTM_layers)],
                                       state_is_tuple=True)
    outputs, states = tf.nn.dynamic_rnn(cell, embed,
                                        sequence_length=_seqlens,
                                        dtype=tf.float32)

#### tags ####
with tf.name_scope("embeddings_tags"):
    embeddings_tags = tf.Variable(
        tf.random_uniform([vocabulary_size_tags,
                           embedding_dimension_tags],
                          -1.0, 1.0), name='embedding_tags')
    embed_tags = tf.nn.embedding_lookup(embeddings_tags, _inputs_tags)

with tf.variable_scope("lstm_tags"):
    # Define a function that gives the output in the right shape
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(hidden_layer_size_tags, forget_bias=1.0)
    cell_tags = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell() for _ in range(num_LSTM_layers_tags)],
                                       state_is_tuple=True)
    outputs_tags, states_tags = tf.nn.dynamic_rnn(cell_tags, embed_tags,
                                        sequence_length=_seqlens,
                                        dtype=tf.float32)
#######

weights = {
     'linear_layer': tf.Variable(tf.truncated_normal([hidden_layer_size + hidden_layer_size_tags, num_classes],
                                                     mean=0, stddev=.01))
 }
biases = {
    'linear_layer': tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))
}
 # extract the last relevant output and use in a linear layer
lstm_states = tf.concat([states[num_LSTM_layers-1][1], states_tags[num_LSTM_layers_tags-1][1]], 1)
final_output = tf.matmul(lstm_states, weights["linear_layer"]) + biases["linear_layer"]

softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output,
                                                  labels=_labels)
cross_entropy = tf.reduce_mean(softmax)

train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(_labels, 1),
                              tf.argmax(final_output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction,
                                   tf.float32)))*100
sample = tf.where(tf.math.logical_not(correct_prediction))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        x_batch, y_batch, seqlen_batch, x2_batch = get_sentence_batch(batch_size,
                                                            train_x, train_y,
                                                            train_seqlens, train_x_tags)
        sess.run(train_step, feed_dict={_inputs: x_batch, _labels: y_batch,
                                        _seqlens: seqlen_batch, _inputs_tags: x2_batch})

        if step % 100 == 0:
            acc = sess.run(accuracy, feed_dict={_inputs: x_batch,
                                                _labels: y_batch,
                                                _seqlens: seqlen_batch, _inputs_tags: x2_batch})
            print("Accuracy at %d: %.5f" % (step, acc))
            samp = sess.run(sample, feed_dict={_inputs: x_batch,
                                                _labels: y_batch,
                                                _seqlens: seqlen_batch, _inputs_tags: x2_batch})
            samp_ind = [row[0] for row in samp]
            x_batch_miss = [x_batch[ind] for ind in samp_ind]
            seqlen_miss = [seqlen_batch[ind] for ind in samp_ind]
            sentences_miss = [[index2word_map[ind] for ind in sent] for sent in x_batch_miss]
            if len(sentences_miss)>0:
                print("Up to 5 miss-classified sample sentences:")
                if len(sentences_miss)<5:
                    n = len(sentences_miss)
                else:
                    n = 5
                for i in range(n):
                    print(" ".join(sentences_miss[i][:seqlen_miss[i]]))
            else:
                print("No miss-classified sentence!")

    mean_acc = 0
    for test_batch in range(5):
        x_test, y_test, seqlen_test, x2_test = get_sentence_batch(batch_size,
                                                         test_x, test_y,
                                                         test_seqlens, test_x_tags)
        batch_pred, batch_acc = sess.run([tf.argmax(final_output, 1), accuracy],
                                         feed_dict={_inputs: x_test,
                                                    _labels: y_test,
                                                    _seqlens: seqlen_test, _inputs_tags: x2_test})
        print("Test batch accuracy %d: %.5f" % (test_batch, batch_acc))
        mean_acc = mean_acc + batch_acc

        samp = sess.run(sample, feed_dict={_inputs: x_batch,
                                           _labels: y_batch,
                                           _seqlens: seqlen_batch, _inputs_tags: x2_batch})
        samp_ind = [row[0] for row in samp]
        x_batch_miss = [x_batch[ind] for ind in samp_ind]
        seqlen_miss = [seqlen_batch[ind] for ind in samp_ind]
        sentences_miss = [[index2word_map[ind] for ind in sent] for sent in x_batch_miss]
        if len(samp) > 0:
            print("Remaining miss-classified sentences:")
            for i in range(len(sentences_miss)):
                print(" ".join(sentences_miss[i][:seqlen_miss[i]]))
        else:
            print("No miss-classified sentence!")
    print("Mean test accuracy: %.5f" % (mean_acc/5))

