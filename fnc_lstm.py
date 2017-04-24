"""Short and sweet LSTM implementation in Tensorflow.

Motivation:
When Tensorflow was released, adding RNNs was a bit of a hack - it required
building separate graphs for every number of timesteps and was a bit obscure
to use. Since then TF devs added things like `dynamic_rnn`, `scan` and `map_fn`.
Currently the APIs are decent, but all the tutorials that I am aware of are not
making the best use of the new APIs.

Advantages of this implementation:
- No need to specify number of timesteps ahead of time. Number of timesteps is
  infered from shape of input tensor. Can use the same graph for multiple
  different numbers of timesteps.
- No need to specify batch size ahead of time. Batch size is infered from shape
  of input tensor. Can use the same graph for multiple different batch sizes.
- Easy to swap out different recurrent gadgets (RNN, LSTM, GRU, your new
  creative idea)
"""


import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
from word_embeddings import WordEmbeddings
from fnc_1_baseline_master.utils.dataset import DataSet
from fnc_1_baseline_master.utils.generate_test_splits import kfold_split, get_stances_for_folds
from fnc_1_baseline_master.feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from fnc_1_baseline_master.feature_engineering import word_overlap_features
from fnc_1_baseline_master.utils.score import report_score, LABELS, score_submission

from fnc_1_baseline_master.utils.system import parse_params, check_version

################################################################################
##                             WORD EMBEDDINGS                                ##
################################################################################
d = DataSet()
folds, hold_out = kfold_split(d, n_folds=512)
fold_stances, hold_out_stances = get_stances_for_folds(d, folds, hold_out)
embeddings = WordEmbeddings()

def get_word_vectors_for_batch(stances, dataset, word_embeddings):
    b, y = [], [] # bodies, true labels
    
    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        #print(stance)
        b.append(dataset.articles[stance['Body ID']])

    embeddings_list = []
    for article in b:
       sentence_embedding = word_embeddings.get_embedding_for_sentence(article)
       embeddings_list.append(sentence_embedding)

    return np.vstack(embeddings_list)
    


################################################################################
##                             GLOBAL FEATURES                                ##
################################################################################

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "fnc_1_baseline_master/features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "fnc_1_baseline_master/features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "fnc_1_baseline_master/features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "fnc_1_baseline_master/features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]

    X = X.reshape(len(y), 44, 1)
    y = np.asarray(y).reshape(len(y), 1, 1)


    return X,y

################################################################################
##                           GRAPH DEFINITION                                 ##
################################################################################

INPUT_SIZE    = 100 #2       # 2 bits per timestep
RNN_HIDDEN    = 10
OUTPUT_SIZE   = 1       # 1 bit per timestep
TINY          = 1e-6    # to avoid NaNs in logs
LEARNING_RATE = 0.01

USE_LSTM = True

inputs  = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))  # (time, batch, in)
outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE)) # (time, batch, out)


## Here cell can be any function you want, provided it has two attributes:
#     - cell.zero_state(batch_size, dtype)- tensor which is an initial value
#                                           for state in __call__
#     - cell.__call__(input, state) - function that given input and previous
#                                     state returns tuple (output, state) where
#                                     state is the state passed to the next
#                                     timestep and output is the tensor used
#                                     for infering the output at timestep. For
#                                     example for LSTM, output is just hidden,
#                                     but state is memory + hidden
# Example LSTM cell with learnable zero_state can be found here:
#    https://gist.github.com/nivwusquorum/160d5cf7e1e82c21fad3ebf04f039317
if USE_LSTM:
    cell = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
else:
    cell = tf.nn.rnn_cell.BasicRNNCell(RNN_HIDDEN)

# Create initial state. Here it is just a constant tensor filled with zeros,
# but in principle it could be a learnable parameter. This is a bit tricky
# to do for LSTM's tuple state, but can be achieved by creating two vector
# Variables, which are then tiled along batch dimension and grouped into tuple.
batch_size    = tf.shape(inputs)[1]
initial_state = cell.zero_state(batch_size, tf.float32)

# Given inputs (time, batch, input_size) outputs a tuple
#  - outputs: (time, batch, output_size)  [do not mistake with OUTPUT_SIZE]
#  - states:  (time, batch, hidden_size)
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)

# project output from rnn output size to OUTPUT_SIZE. Sometimes it is worth adding
# an extra layer here.
final_projection = lambda x: layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)

# apply projection to every timestep.
predicted_outputs = tf.map_fn(final_projection, rnn_outputs)

# compute elementwise cross entropy.
error = -(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY))
error = tf.reduce_mean(error)

# optimize
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)

# assuming that absolute difference between output and correct answer is 0.5
# or less we can round it to the correct output.
accuracy = tf.reduce_mean(tf.cast(tf.abs(outputs - predicted_outputs) < 0.5, tf.float32))

print ("Finished defining graph")

################################################################################
##                           TRAINING LOOP                                    ##
################################################################################

NUM_BITS = 10
ITERATIONS_PER_EPOCH = 100
# BATCH_SIZE = 16


# acquire data from files
#d = DataSet()
# folds, hold_out = kfold_split(d, n_folds=10)
# fold_stances, hold_out_stances = get_stances_for_folds(d, folds, hold_out)

x_vals = {}
y_vals = {}

for fold in fold_stances:
  x_vals[fold], y_vals[fold] = get_word_vectors_for_batch(fold_stances[fold], d, embeddings) # generate_features(fold_stances[fold], d, str(fold))

valid_x, valid_y = get_word_vectors_for_batch(hold_out_stances, d, embeddings) # generate_features(hold_out_stances, d, "holdout")

print ("Finished separating folds")

session = tf.Session()
# For some reason it is our job to do this:
session.run(tf.initialize_all_variables())

for epoch in range(1000):
    epoch_error = 0
    for fold in fold_stances: # for _ in range(ITERATIONS_PER_EPOCH):
        # here train_fn is what triggers backprop. error and accuracy on their
        # own do not trigger the backprop.
        # x, y = generate_batch(num_bits=NUM_BITS, batch_size=BATCH_SIZE)
        # TODO replace above line with getting feature vectors for current batch
        x = x_vals[fold]
        y = y_vals[fold]

        epoch_error += session.run([error, train_fn], {
            inputs: x,
            outputs: y,
        })[0]

    epoch_error /= len(fold_stances) # ITERATIONS_PER_EPOCH
    valid_accuracy = session.run(accuracy, {
        inputs:  valid_x,
        outputs: valid_y,
    })
    print ("Epoch %d, train error: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_error, valid_accuracy * 100.0))
