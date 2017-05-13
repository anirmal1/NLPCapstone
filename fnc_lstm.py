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
from sklearn import metrics
from fnc_1_baseline_master.utils.dataset import DataSet
from fnc_1_baseline_master.utils.generate_test_splits import kfold_split, get_stances_for_folds
from fnc_1_baseline_master.feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from fnc_1_baseline_master.feature_engineering import word_overlap_features
from fnc_1_baseline_master.utils.score import report_score, LABELS, score_submission
from fnc_1_baseline_master.utils.system import parse_params, check_version

################################################################################
##                             WORD EMBEDDINGS                                ##
################################################################################
def get_articles_word_vectors(stances, dataset, word_embeddings):
	b, y, h = [], [], []  # bodies, true labels, headlines
	for stance in stances:
		#y.append([[LABELS.index(stance['Stance'])]])
		zeros = np.zeros(4)
		zeros[LABELS.index(stance['Stance'])] = 1
		y.append(np.array([zeros]))
		h.append(stance['Headline'])
		b.append(dataset.articles[stance['Body ID']])

	embeddings_list = []
	headline_embeddings_list = []
	ys = []
	for i in range(len(b)):
		article = word_embeddings.get_embedding_for_sentence(b[i])
		headline = word_embeddings.get_embedding_for_headline(h[i]) # trying to get equal length headline #get_embedding_for_sentence(h[i])
		y_val = y[i]
		if article.ndim == 2 and headline.ndim == 2 and headline.shape[0] == 30 and headline.shape[1] == 100:
			embeddings_list.append(article)
			headline_embeddings_list.append(headline)
			ys.append(y_val)

	return np.array(headline_embeddings_list), np.array(embeddings_list), np.array(ys, dtype=np.float32) # TODO how do you properly return the embeddings in 3 dimensions?? :(
    


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
    X_discuss = gen_or_load_feats(discuss_features, h, b, "fnc_1_baseline_master/features/discuss."+name+".npy")
    X_vader_sentiment = gen_or_load_feats(get_sentiment_difference, h, b, "fnc_1_baseline_master/features/vader_sentiment."+name+".npy")
    X_tfidf_headline, X_tfidf_bodies = gen_or_load_feats(get_sentiment_difference, h, b, "fnc_1_baseline_master/features/tfidf."+name+".npy")
 
    X = np.c_[X_tfidf_headline, X_tfidf_bodies, X_vader_sentiment, X_discuss, X_hand, X_polarity, X_refuting, X_overlap]

    X = X.reshape(len(y), 44, 1)
    y = np.asarray(y).reshape(len(y), 1, 1)

    return X,y

################################################################################
##                           GRAPH DEFINITION                                 ##
################################################################################

INPUT_SIZE    = 100 # Length of GLoVe word embeddings (100d)
RNN_HIDDEN    = 10
OUTPUT_SIZE   = 4       # Final output (label)
HIDDEN_OUTPUT_SIZE = 4 # Softmax over all four labels
TINY          = 1e-6    # to avoid NaNs in logs
LEARNING_RATE = 0.01

USE_LSTM = True

inputs_articles = tf.placeholder(tf.float32, (None, 200, INPUT_SIZE))  # (batch, time, in)
inputs_headlines = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))

outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE)) # (batch, time, out)

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
    #cell_articles = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
    #cell_headlines = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True) 
	with tf.variable_scope('scope1') as scope1:  
		# Create cell
		cell_articles = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
		# Initialize batch size, initial states
		batch_size_articles= tf.shape(inputs_articles)[0]
		initial_state_articles = cell_articles.zero_state(batch_size_articles, tf.float32)
		# Hidden states, outputs
		rnn_outputs_articles, rnn_states_articles = tf.nn.dynamic_rnn(cell_articles, inputs_articles, initial_state=initial_state_articles, time_major=False)
	with tf.variable_scope('scope1') as scope1:
		scope1.reuse_variables() 
		# Create cell
		cell_headlines = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True, reuse=True) 
		# Initialize batch size, initial states
		batch_size_headlines= tf.shape(inputs_headlines)[0]
		initial_state_headlines = rnn_states_articles 
	        # Hidden states, outputs
		rnn_outputs_headlines, rnn_states_headlines = tf.nn.dynamic_rnn(cell_headlines, inputs_headlines, initial_state=initial_state_headlines, time_major=False)
else:
	cell = tf.nn.rnn_cell.BasicRNNCell(RNN_HIDDEN)

# Create initial state. Here it is just a constant tensor filled with zeros,
# but in principle it could be a learnable parameter. This is a bit tricky
# to do for LSTM's tuple state, but can be achieved by creating two vector
# Variables, which are then tiled along batch dimension and grouped into tuple.
'''batch_size_articles= tf.shape(inputs_articles)[0]
initial_state_articles = cell_articles.zero_state(batch_size_articles, tf.float32)
batch_size_headlines= tf.shape(inputs_headlines)[0]
initial_state_headlines = cell_headlines.zero_state(batch_size_headlines, tf.float32)'''

# Given inputs (time, batch, input_size) outputs a tuple
#  - outputs: (time, batch, output_size)  [do not mistake with OUTPUT_SIZE]
#  - states:  (time, batch, hidden_size)
'''rnn_outputs_articles, rnn_states_articles = tf.nn.dynamic_rnn(cell_articles, inputs_articles, initial_state=initial_state_articles, time_major=False)
rnn_outputs_headlines, rnn_states_headlines = tf.nn.dynamic_rnn(cell_headlines, inputs_headlines, initial_state=initial_state_headlines, time_major=False)'''

# Concatenate articles and headlines rnn_outputs
rnn_outputs = tf.concat([rnn_outputs_articles, rnn_outputs_headlines], 1)

# Concatenate global features to rnn_outputs 

# project output from rnn output size to OUTPUT_SIZE. Sometimes it is worth adding
# an extra layer here.
final_projection = lambda x: layers.linear(x, num_outputs=HIDDEN_OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)

# apply projection to every timestep.
predicted_outputs = tf.map_fn(final_projection, rnn_outputs)

# Take softmax, Get predicted label
softmaxes = tf.nn.softmax(predicted_outputs[0:, -1, 0:]) #tf.nn.softmax(predicted_outputs)
pred_stance = tf.argmax(softmaxes, 1)
#ipred_array = np.zeros(4)
#pred_array[LABELS.index(pred_stance)] = 1

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

# acquire data from files
# NOTE: For the word vectors, folds really means batches
d = DataSet()
folds, hold_out = kfold_split(d, n_folds=512)
fold_stances, hold_out_stances = get_stances_for_folds(d, folds, hold_out)
embeddings = WordEmbeddings()

x_articles = {}
x_headlines = {}
y_vals = {}

for fold in fold_stances:
	x_headlines[fold], x_articles[fold], y_vals[fold] = get_articles_word_vectors(fold_stances[fold], d, embeddings) 

valid_x_headlines, valid_x_articles, valid_y = get_articles_word_vectors(hold_out_stances, d, embeddings) 

print ("Finished separating batches")

session = tf.Session()
# For some reason it is our job to do this:
session.run(tf.global_variables_initializer())

for fold in fold_stances: #for epoch in range(10):
	epoch_error = 0
	for epoch in range(10):# for fold in fold_stances:
		ids = list(range(len(folds)))
		del ids[fold]
		x_train_articles = np.vstack(tuple([x_articles[i] for i in ids]))
		x_train_headlines = np.vstack(tuple([x_headlines[i] for i in ids]))
		y_train = np.vstack(tuple([y_vals[i] for i in ids]))

		x_article_batch = x_articles[fold]
		x_headline_batch = x_headlines[fold]
		y = y_vals[fold]

		# Training error
		epoch_error += session.run([error, train_fn], {
			inputs_articles: x_train_articles, # x_article_batch,
			inputs_headlines: x_train_headlines, # x_headline_batch,
			outputs: y_train # y
		})[0]

		print("Epoch " + str(epoch) + " error: " + str(epoch_error/len(fold_stances)))

	# Test error
	valid_accuracy, pred_y_stances = session.run([accuracy, pred_stance], {
		inputs_articles:  x_article_batch, # valid_x_articles,
		inputs_headlines: x_headline_batch, # valid_x_headlines
		outputs: y # valid_y
	})

	simple_y = np.array([array[0].tolist().index(1) for array in y])
	'''
		print('True outputs: ' + str(simple_y))
		print('Shape of true outputs: '+ str(simple_y.shape))
		print('Type of true outputs: ' + str(simple_y.dtype))
		print('Predicted outputs: ' + str(pred_y_stances))
		print('Shape of predicted outputs: '+ str(pred_y_stances.shape))
		print('Type of predicted outputs: ' + str(pred_y_stances.dtype))
		print ("Epoch %d, train error: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_error, valid_accuracy * 100.0))
	'''	

	f1_score = metrics.f1_score(simple_y, pred_y_stances, average='macro')
	print("F1 MEAN score: " + str(f1_score))
	f1_score_labels =  metrics.f1_score(simple_y, pred_y_stances, labels=[0, 1, 2, 3], average=None)
	print("F1 LABEL scores: " + str(f1_score_labels))

	# Convert to string labels for FNC scoring metric
	label_map = {0 : "agree", 1 : "disagree", 2 : "discuss", 3 : "unrelated"}
	simple_y_str = [label_map[label] for label in simple_y]
	pred_y_stances_str = [label_map[label] for label in pred_y_stances]
	report_score(simple_y_str, pred_y_stances_str)

valid_accuracy, pred_y_stances = session.run([accuracy, pred_stance], {
		inputs_articles:  valid_x_articles,
		inputs_headlines: valid_x_headlines,
		outputs: valid_y
	})

simple_y = np.array([array[0].tolist().index(1) for array in valid_y])
f1_score = metrics.f1_score(simple_y, pred_y_stances, average='macro')
print("F1 MEAN score: " + str(f1_score))
f1_score_labels =  metrics.f1_score(simple_y, pred_y_stances, labels=[0, 1, 2, 3], average=None)
print("F1 LABEL scores: " + str(f1_score_labels))

# Convert to string labels for FNC scoring metric
label_map = {0 : "agree", 1 : "disagree", 2 : "discuss", 3 : "unrelated"}
simple_y_str = [label_map[label] for label in simple_y]
pred_y_stances_str = [label_map[label] for label in pred_y_stances]
report_score(simple_y_str, pred_y_stances_str)
       
