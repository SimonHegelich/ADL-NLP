# -*- coding: utf-8 -*-


# https://gist.github.com/mikalv/3947ccf21366669ac06a01f39d7cff05
# http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
import tensorflow as tf
import numpy as np
import os, sys
import re
import collections

#set hyperparameters
max_len = 40
step = 10
num_units = 128
learning_rate = 0.001
batch_size = 200
epoch = 50
temperature = 0.8
SAVE_PATH = '/home/simon/Firma/AI/HegelMachine/HegelPython/logs/TextGen/Rilke/'


if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

def tokens(text):
    """
    Get all words from corpus
    """
    text = re.sub(r'[0-9]+', '', text)
    return re.findall(r'\w+', text.lower())

WORDS = tokens(file('RilkeBig.txt').read())
WORD_COUNTS = collections.Counter(WORDS)

def edits0(word):
    """
    Return all strings that are zero edits away (i.e. the word itself).
    """
    return{word}

def edits1(word):
    """
    Return all strings that are one edits away.
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyzäüö'
    def splits(word):
        """
        return a list of all possible pairs
        that the input word is made of
        """
        return [(word[:i], word[i:]) for i in range(len(word)+1)]
    pairs = splits(word)
    deletes = [a+b[1:] for (a,b) in pairs if b]
    transposes = [a+b[1]+b[0]+b[2:] for (a,b) in pairs if len(b) >1]
    replaces = [a+c+b[1:] for (a,b) in pairs for c in alphabet if b]
    inserts = [a+c+b for (a,b) in pairs for c in alphabet]
    return(set(deletes + transposes + replaces + inserts))

def edits2(word):
    """
    return all strings that are two edits away.
    """
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}

def known(words):
    return {w for w in words if w in WORD_COUNTS}

def correct(word):
    candidates = (known(edits0(word)) or
                 known(edits1(word)) or
                 known(edits2(word)) or
                 [word])
    return max(candidates, key=WORD_COUNTS.get)

def correct_match(match):#
    """
    spell-correct word in match,
    and perserve upper/lower/title case
    """
    word = match.group()
    def case_of(text):
        return(str.upper if text.isupper() else
              str.lower if text.islower() else
              str.title if text.istitle() else
              str)
    return case_of(word)(correct(word.lower()))

def correct_text_generic(text):
    """
    correct all words in text
    """
    return re.sub('[a-zA-Z]+', correct_match, text)

def read_data(file_name):
    '''
     open and read text file
    '''
    text = open(file_name, 'r').read()
    return text.lower()

def featurize(text):
    '''
     featurize the text to train and target dataset
    '''
    unique_chars = list(set(text))
    len_unique_chars = len(unique_chars)

    input_chars = []
    output_char = []

    for i in range(0, len(text) - max_len, step):
        input_chars.append(text[i:i+max_len])
        output_char.append(text[i+max_len])

    train_data = np.zeros((len(input_chars), max_len, len_unique_chars))
    target_data = np.zeros((len(input_chars), len_unique_chars))

    for i , each in enumerate(input_chars):
        for j, char in enumerate(each):
            train_data[i, j, unique_chars.index(char)] = 1
        target_data[i, unique_chars.index(output_char[i])] = 1
    return train_data, target_data, unique_chars, len_unique_chars

def rnn(x, weight, bias, len_unique_chars):
    '''
     define rnn cell and prediction
    '''
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, len_unique_chars])
    x = tf.split(x, max_len, 0)

    cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
    outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)
    prediction = tf.matmul(outputs[-1], weight) + bias
    return prediction

def sample(predicted):
    '''
     helper function to sample an index from a probability array
    '''
    exp_predicted = np.exp(predicted/temperature)
    predicted = exp_predicted / np.sum(exp_predicted)
    probabilities = np.random.multinomial(1, predicted, 1)
    return probabilities

def run(train_data, target_data, unique_chars, len_unique_chars):
    '''
     main run function
    '''
    x = tf.placeholder("float", [None, max_len, len_unique_chars], name ="Input")
    y = tf.placeholder("float", [None, len_unique_chars], name = "Output")
    weight = tf.Variable(tf.random_normal([num_units, len_unique_chars]))
    bias = tf.Variable(tf.random_normal([len_unique_chars]))

    prediction = rnn(x, weight, bias, len_unique_chars)
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    cost = tf.reduce_mean(softmax)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)


    import glob
    if glob.glob(SAVE_PATH + '*.meta'):
        tf.reset_default_graph()
        imported_meta = tf.train.import_meta_graph(glob.glob(SAVE_PATH + '*.meta')[0])
        sess=tf.Session()
        imported_meta.restore(sess, tf.train.latest_checkpoint(SAVE_PATH))
        print (" restoring an old model and training it further ")

        x = sess.graph.get_tensor_by_name("Input:0")
        y = sess.graph.get_tensor_by_name("Output:0")
        prediction = tf.get_collection("prediction")[0]
        optimizer = tf.get_collection("optimizer")[0]

    else:

        init_op = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_op)
        print("Building model from scratch!")

    saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=1, save_relative_paths=True)
    num_batches = int(len(train_data)/batch_size)

    for i in range(epoch):
        print "----------- Epoch {0}/{1} -----------".format(i+1, epoch)
        count = 0
        for _ in range(num_batches):
            train_batch, target_batch = train_data[count:count+batch_size], target_data[count:count+batch_size]
            count += batch_size
            sess.run([optimizer] ,feed_dict={x:train_batch, y:target_batch})
            tf.add_to_collection("optimizer", optimizer)

        #get on of training set as seed
        seed = train_batch[:1:]

        #to print the seed 40 characters
        seed_chars = ''
        for each in seed[0]:
                seed_chars += unique_chars[np.where(each == max(each))[0][0]]
        print "Seed:", seed_chars

        #predict next 500 characters
        for i in range(500):
            if i > 0:
                remove_fist_char = seed[:,1:,:]
                seed = np.append(remove_fist_char, np.reshape(probabilities, [1, 1, len_unique_chars]), axis=1)
            predicted = sess.run([prediction], feed_dict = {x:seed})
            tf.add_to_collection("prediction", prediction)
            predicted = np.asarray(predicted[0]).astype('float64')[0]
            probabilities = sample(predicted)
            predicted_chars = unique_chars[np.argmax(probabilities)]
            seed_chars += predicted_chars
        print 'Result:', seed_chars
        print 'Corrected:', correct_text_generic(seed_chars)
    ui = True
    while ui == True:
        seed_chars = raw_input("Enter a seed: ")
        for i in range(280):
            if i > 0:
                remove_fist_char = seed[:, 1:, :]
                seed = np.append(remove_fist_char, np.reshape(probabilities, [1, 1, len_unique_chars]), axis=1)
            predicted = sess.run([prediction], feed_dict={x: seed})
            predicted = np.asarray(predicted[0]).astype('float64')[0]
            probabilities = sample(predicted)
            predicted_chars = unique_chars[np.argmax(probabilities)]
            seed_chars += predicted_chars
        # print 'Result:', seed_chars
        print 'Corrected:', correct_text_generic(seed_chars)
        action = raw_input("Do you want to try another seed? (yes=y, no=n)?: ")
        if action != "y":
            ui = False
    save_path = saver.save(sess, SAVE_PATH, global_step=10)
    print("Model saved in file: %s" % save_path)
    sess.close()

    tf.reset_default_graph()

if __name__ == "__main__":
    text = read_data('RilkeLyrik.txt')
    text = re.sub(r'[0-9]+', '', text)
    train_data, target_data, unique_chars, len_unique_chars = featurize(text)
    tf.reset_default_graph()
    run(train_data, target_data, unique_chars, len_unique_chars)
