# based in parts on https://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/
# and https://adventuresinmachinelearning.com/word2vec-keras-tutorial/
# rewritten for tensorflow 2.2 integrated subword tokenizer

import tensorflow as tf
import sentencepiece as spm
import collections
import numpy as np
import io

# parameters
filename = "/home/simon/Downloads/poems_processed.txt" # Any big txt...
vocab_size=10000
sub_size=15000
window_size = 3
vector_dim = 768
epochs = 1000000

valid_size = 8     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

with open(filename, "r", encoding = "utf8") as f: text = f.read()

# functions

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


spm.SentencePieceTrainer.Train('--input={} --model_prefix=m --vocab_size={}'.format(filename, sub_size))

sp = spm.SentencePieceProcessor()
sp.load('m.model')

print(sp.encode_as_pieces('this is a test.'))

vocabulary = sp.encode_as_pieces(text)

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocab_size)


sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
couples, labels = tf.keras.preprocessing.sequence.skipgrams(data, vocab_size, window_size=window_size, sampling_table=sampling_table)
word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

print(couples[:10], labels[:10])

# create some input variables
input_target = tf.keras.Input((1,))
input_context = tf.keras.Input((1,))

embedding = tf.keras.layers.Embedding(vocab_size, vector_dim, input_length=1, name='embedding')

target = embedding(input_target)
target = tf.keras.layers.Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = tf.keras.layers.Reshape((vector_dim, 1))(context)

# setup a cosine similarity operation which will be output in a secondary model
similarity = tf.keras.layers.dot(inputs=[target, context], axes=1, normalize=True)

# now perform the dot product operation to get a similarity measure
dot_product = tf.keras.layers.dot(inputs=[target, context], axes=1)
dot_product = tf.keras.layers.Reshape((1,))(dot_product)
# add the sigmoid output layer
output = tf.keras.layers.Dense(1, activation='sigmoid')(dot_product)

# create the primary training model
model = tf.keras.Model(inputs=[input_target, input_context], outputs=output)
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

# create a secondary validation model to run our similarity checks during training
validation_model = tf.keras.Model(inputs=[input_target, input_context], outputs=similarity)

class SimilarityCallback:
    def run_sim(self):
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 4  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        for i in range(vocab_size):
            in_arr1[0,] = valid_word_idx
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim
sim_cb = SimilarityCallback()

arr_1 = np.zeros((1,))
arr_2 = np.zeros((1,))
arr_3 = np.zeros((1,))
for cnt in range(epochs):
    idx = np.random.randint(0, len(labels)-1)
    arr_1[0,] = word_target[idx]
    arr_2[0,] = word_context[idx]
    arr_3[0,] = labels[idx]
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
    if cnt % 100 == 0:
        print("Iteration {}, loss={}".format(cnt, loss))
    if cnt % 10000 == 0:
        sim_cb.run_sim() # For huge datasets this is very time consuming. 

e = model.layers[2]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(vocab_size):
  word = reverse_dictionary[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

# load vecs.tsv and meta.tsv in projector.tensorflow.org

print("stop")