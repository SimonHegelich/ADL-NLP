import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import gutenberg

alice = gutenberg.raw(fileids='carroll-alice.txt')
default_st = nltk.sent_tokenize
alice_sentences = default_st(text=alice)
default_wt = nltk.word_tokenize
alice_words = [default_wt(sentence) for sentence in alice_sentences]
print(len(alice_sentences))
len_alice = len(alice_sentences)

hamlet = gutenberg.raw(fileids='shakespeare-hamlet.txt')
hamlet_sentences = default_st(text=hamlet)
hamlet_words = [default_wt(sentence) for sentence in hamlet_sentences]
print(len(hamlet_sentences))
len_hamlet = len(hamlet_sentences)

data = alice_words + hamlet_words
len_data = len(data)

max_len = 0
for sent in data:
    if len(sent)>max_len:
        max_len = len(sent)

batch_size = 128
embedding_dimension = 64
num_classes = 2
hidden_layer_size = 32
#element_size = 1

seqlens = []

for sentence_id in range(len_data):
    seqlens.append(len(data[sentence_id]))

    if len(data[sentence_id]) < max_len:
        pads = ["PAD"]*(max_len-len(data[sentence_id]))
        data[sentence_id] = data[sentence_id] + pads

# seqlens *= 2
labels = [1] * len_alice + [0] * len_hamlet
for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0]*2
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

data_indices = list(range(len(data)))
np.random.shuffle(data_indices)
data = np.array(data)[data_indices]
labels = np.array(labels)[data_indices]
seqlens = np.array(seqlens)[data_indices]
train_x = data[:(len(data_indices)/2)]
train_y = labels[:(len(data_indices)/2)]
train_seqlens = seqlens[:(len(data_indices)/2)]

test_x = data[(len(data_indices)/2):]
test_y = labels[(len(data_indices)/2):]
test_seqlens = seqlens[(len(data_indices)/2):]


def get_sentence_batch(batch_size, data_x,
                       data_y, data_seqlens):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [[word2index_map[word] for word in data_x[i]]
         for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x, y, seqlens


_inputs = tf.placeholder(tf.int32, shape=[batch_size, max_len])
_labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
# seqlens for dynamic calculation
_seqlens = tf.placeholder(tf.int32, shape=[batch_size])

with tf.name_scope("embeddings"):
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size,
                           embedding_dimension],
                          -1.0, 1.0), name='embedding')
    embed = tf.nn.embedding_lookup(embeddings, _inputs)


num_LSTM_layers = 2
with tf.variable_scope("lstm"):
    # Define a function that gives the output in the right shape
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, forget_bias=1.0)
    cell = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell() for _ in range(num_LSTM_layers)],
                                       state_is_tuple=True)
    outputs, states = tf.nn.dynamic_rnn(cell, embed,
                                        sequence_length = _seqlens,
                                        dtype=tf.float32)

weights = {
     'linear_layer': tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes],
                                                     mean=0, stddev=.01))
 }
biases = {
    'linear_layer': tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))
}
 # extract the last relevant output and use in a linear layer
final_output = tf.matmul(states[num_LSTM_layers-1][1],
                         weights["linear_layer"]) + biases["linear_layer"]

softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output,
                                                  labels=_labels)
cross_entropy = tf.reduce_mean(softmax)

train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(_labels, 1),
                              tf.argmax(final_output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction,
                                   tf.float32)))*100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size,
                                                            train_x, train_y,
                                                            train_seqlens)
        sess.run(train_step, feed_dict={_inputs: x_batch, _labels: y_batch,
                                        _seqlens: seqlen_batch})

        if step % 100 == 0:
            acc = sess.run(accuracy, feed_dict={_inputs: x_batch,
                                                _labels: y_batch,
                                                _seqlens: seqlen_batch})
            print("Accuracy at %d: %.5f" % (step, acc))

    for test_batch in range(5):
        x_test, y_test, seqlen_test = get_sentence_batch(batch_size,
                                                         test_x, test_y,
                                                         test_seqlens)
        batch_pred, batch_acc = sess.run([tf.argmax(final_output, 1), accuracy],
                                         feed_dict={_inputs: x_test,
                                                    _labels: y_test,
                                                    _seqlens: seqlen_test})
        print("Test batch accuracy %d: %.5f" % (test_batch, batch_acc))

