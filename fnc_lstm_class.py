import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
from word_embeddings import WordEmbeddings
from sklearn import metrics
from fnc_1_baseline_master.utils.dataset import DataSet
from fnc_1_baseline_master.utils.generate_test_splits import kfold_split, get_stances_for_folds
from fnc_1_baseline_master.feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats, word_overlap_features, discuss_features, get_sentiment_difference, get_tfidf
from fnc_1_baseline_master.utils.score import report_score, LABELS, score_submission
from fnc_1_baseline_master.utils.system import parse_params, check_version

model_path = 'lstm_model.ckpt' # for saving the model later

INPUT_SIZE = 100 # length of GLoVe word embeddings
RNN_HIDDEN = 100
OUTPUT_SIZE = 4
HIDDEN_OUTPUT_SIZE = 4
TINY = 1e-6
LEARNING_RATE = 0.0001
BATCH_SIZE = 1024

################################################################################
##                           BIDIRECTIONAL LSTM                               ##
################################################################################

class Classifier(object):
	def __init__(self):
		self.session = tf.Session()

		# input/output placeholders
		self.inputs_articles = tf.placeholder(tf.float32, (None, 200, INPUT_SIZE), name='input_articles')
		self.inputs_headlines = tf.placeholder(tf.float32, (None, 30, INPUT_SIZE), name='inputs_headlines')	
		self.outputs = tf.placeholder(tf.float32, (None, OUTPUT_SIZE), name='outputs') # TODO change to two dimensions
		self.h_lengths = tf.placeholder(tf.int32, (None, 2))
		self.a_lengths = tf.placeholder(tf.int32, (None, 2))
		self.global_feats = tf.placeholder(tf.float32, (None, 44))

		# LSTM cells, TODO make these bidrectional!
		with tf.variable_scope('scope1') as scope1:  
			# Create cell
			self.cell_articles_fw = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
			self.cell_articles_fw = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(self.cell_articles_fw, input_keep_prob=0.7, output_keep_prob=0.2)
			self.cell_articles_bw = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
			self.cell_articles_bw = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(self.cell_articles_bw, input_keep_prob=0.7, output_keep_prob=0.2)
			self.rnn_outputs_articles, self.rnn_states_articles = 	tf.nn.bidirectional_dynamic_rnn(self.cell_articles_fw, self.cell_articles_bw, self.inputs_articles, dtype=tf.float32)
			# Initialize batch size, initial states
			'''
			batch_size_articles= tf.shape(self.inputs_articles)[0]
			initial_state_articles = self.cell_articles.zero_state(batch_size_articles, tf.float32)
			# Hidden states, outputs
			self.rnn_outputs_articles, self.rnn_states_articles = tf.nn.dynamic_rnn(self.cell_articles, self.inputs_articles, initial_state=initial_state_articles, time_major=False)
			'''
		with tf.variable_scope('scope1') as scope1:
			scope1.reuse_variables() 
			# Create cell
			self.cell_headlines_fw = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True, reuse=True)
			self.cell_headlines_fw = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(self.cell_headlines_fw, input_keep_prob=0.7, output_keep_prob=0.2) 
			self.cell_headlines_bw = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True, reuse=True)
			self.cell_headlines_bw = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(self.cell_headlines_bw, input_keep_prob=0.7, output_keep_prob=0.2) 
			self.rnn_outputs_headlines, self.rnn_states_headlines = tf.nn.bidirectional_dynamic_rnn(self.cell_headlines_fw, self.cell_headlines_bw, self.inputs_headlines, dtype=tf.float32)
			'''
			# Initialize batch size, initial states
			batch_size_headlines= tf.shape(self.inputs_headlines)[0]
			initial_state_headlines = self.rnn_states_articles 
			# Hidden states, outputs
			self.rnn_outputs_headlines, self.rnn_states_headlines = tf.nn.dynamic_rnn(self.cell_headlines, self.inputs_headlines, initial_state=initial_state_headlines, time_major=False)
			'''
		# make prediction
		out1 = tf.gather_nd(self.rnn_outputs_articles[0], self.a_lengths)
		out2 = tf.gather_nd(self.rnn_outputs_articles[1], self.a_lengths)
		out3 = tf.gather_nd(self.rnn_outputs_headlines[0], self.h_lengths)
		out4 = tf.gather_nd(self.rnn_outputs_headlines[1], self.h_lengths)

		self.rnn_outputs = tf.concat([out1, out2, out3, out4, self.global_feats], 1)
		# self.rnn_outputs = tf.concat([self.rnn_outputs_articles[0], self.rnn_outputs_articles[1], self.rnn_outputs_headlines[0], self.rnn_outputs_headlines[1]], 1)
		self.final_projection = layers.fully_connected(self.rnn_outputs, num_outputs=HIDDEN_OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)
		self.pred_stance = tf.argmax(self.final_projection, 1)
		#self.softmaxes = tf.nn.softmax(final_projection[0:, 0:])

		# cross entropy loss TODO compute cross entropy between softmax and expected output (a one-hot vector)
		#self.error = -(self.outputs * tf.log(self.softmaxes + TINY) + (1.0 - self.outputs) * tf.log(1.0 - self.softmaxes + TINY))
		# self.error = -(self.outputs * tf.log(predicted_outputs + TINY) + (1.0 - self.outputs) * tf.log(1.0 - predicted_outputs + TINY))
		#self.error = tf.reduce_mean(self.error)
		self.error = tf.nn.softmax_cross_entropy_with_logits(labels=self.outputs, logits=self.final_projection)
		self.error = tf.reduce_mean(self.error)
		self.train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, name='train_fn').minimize(self.error)
		
		# accuracy TODO what is this even doing...
		#self.accuracy = tf.reduce_mean(tf.cast(tf.abs(self.outputs - final_projection) < 0.5, tf.float32))


################################################################################
##                             WORD VECTORS                                   ##
################################################################################

def get_articles_word_vectors(stances, dataset, word_embeddings):
	b, y, h = [], [], []  # bodies, true labels, headlines
	length_h, length_a = [], []
	j = 0
	for stance in stances:
		#y.append([[LABELS.index(stance['Stance'])]])
		zeros = np.zeros(4)
		zeros[LABELS.index(stance['Stance'])] = 1
		y.append(np.array(zeros))
		h.append(stance['Headline'])
		b.append(dataset.articles[stance['Body ID']])
		length_a.append(np.array([j % BATCH_SIZE, dataset.lengths[stance['Body ID']] - 1]))
		j += 1

	embeddings_list = []
	headline_embeddings_list = []
	ys = []
	for i in range(len(b)):
		article = word_embeddings.get_embedding_for_sentence(b[i])
		headline, length = word_embeddings.get_embedding_for_headline(h[i]) # trying to get equal length headline #get_embedding_for_sentence(h[i])
		y_val = y[i]
		length_h.append(np.array([i % BATCH_SIZE, length - 1]))

		embeddings_list.append(article)
		headline_embeddings_list.append(headline)
		ys.append(y_val)

	return np.array(headline_embeddings_list), np.array(embeddings_list), np.array(ys, dtype=np.float32), np.array(length_h), np.array(length_a)  # TODO return outputs in one-hot vector form (2 dimensions)

################################################################################
##                             GLOBAL FEATURES                                ##
################################################################################

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])
    
    # Transform b to be one string instead of a list of words
    b = [" ".join(body) for body in b]

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "fnc_1_baseline_master/features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "fnc_1_baseline_master/features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "fnc_1_baseline_master/features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "fnc_1_baseline_master/features/hand."+name+".npy")
    X_discuss = gen_or_load_feats(discuss_features, h, b, "fnc_1_baseline_master/features/discuss."+name+".npy")
    X_vader_sentiment = gen_or_load_feats(get_sentiment_difference, h, b, "fnc_1_baseline_master/features/vader_sentiment."+name+".npy")
    X_tfidf_headline, X_tfidf_bodies = gen_or_load_feats(get_tfidf, h, b, "fnc_1_baseline_master/features/tfidf."+name+".npy")
    #print(X_hand.shape)
    print("X_discuss: " + str(X_discuss.shape))
    print("X_vader: " + str(X_vader_sentiment.shape))
    print("X_tfidf_h: " + str(X_tfidf_headline.shape))
    print("X_tfidf_b: " + str(X_tfidf_bodies.shape))

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    print("X: " + X.shape)
    #X = X.reshape(len(y), 44, 1)
    #y = np.asarray(y).reshape(len(y), 1, 1)

    return X,y
 
################################################################################
##                                 BATCHING                                   ##
################################################################################
def create_batches(x_articles, x_headlines, y, lengths_h, lengths_a, global_feats):	
	article_batches = []
	headline_batches = []
	output_batches = []
	length_h_batches = []
	length_a_batches = []
	global_batches = []

	batch_size = BATCH_SIZE
	start = 0
	while start < len(x_articles):
		article_chunk = x_articles[start:start + batch_size]
		headline_chunk = x_headlines[start:start + batch_size]
		output_chunk = y[start:start + batch_size]
		length_h_chunk = lengths_h[start:start + batch_size]
		length_a_chunk = lengths_a[start:start + batch_size]
		for i in range(0, len(length_h_chunk)):
			length_h_chunk[i][0] = i
			length_a_chunk[i][0] = i
		global_batches_chunk = global_feats[start:start + batch_size]
		article_batches.append(article_chunk)
		headline_batches.append(headline_chunk)
		output_batches.append(output_chunk)
		length_h_batches.append(length_h_chunk)
		length_a_batches.append(length_a_chunk)
		global_batches.append(global_batches_chunk)
		start += batch_size
	
	return article_batches, headline_batches, output_batches, length_h_batches, length_a_batches, global_batches

################################################################################
##                               TRAINING LOOP                                ##
################################################################################

def main():
	d = DataSet()
	folds, hold_out = kfold_split(d, n_folds=10)
	fold_stances, hold_out_stances = get_stances_for_folds(d, folds, hold_out)
	embeddings = WordEmbeddings()
	print('Created data set and word embeddings')
	
	# create classifier
	model = Classifier()
	print('Set up model')

	# get word vector data
	x_articles = {}
	x_headlines = {}
	y_vals = {}
	lengths_a = {}
	lengths_h = {}

	x_global = {}
	y_global = {}
	
	for fold in fold_stances:
		x_headlines[fold], x_articles[fold], y_vals[fold], lengths_h[fold], lengths_a[fold] = get_articles_word_vectors(fold_stances[fold], d, embeddings)
		x_global[fold], y_global[fold] = generate_features(fold_stances[fold], d, str(fold))

	test_x_headlines, test_x_articles, test_y, test_h_lengths, test_a_lengths = get_articles_word_vectors(hold_out_stances, d, embeddings)
	test_x_global, test_y_global = generate_features(hold_out_stances, d, 'holdout')
	print('Finished separating folds')

	# TODO get global feature data

	# train LSTM (fold -> epoch -> batch)
	model.session.run(tf.global_variables_initializer())

	for fold in fold_stances:
		ids = list(range(len(folds)))
		del ids[fold]
		x_train_articles = np.vstack(tuple([x_articles[i] for i in ids]))
		x_train_headlines = np.vstack(tuple([x_headlines[i] for i in ids]))
		y_train = np.vstack(tuple([y_vals[i] for i in ids]))
		lengths_h_train = np.vstack(tuple([lengths_h[i] for i in ids]))
		lengths_a_train = np.vstack(tuple([lengths_a[i] for i in ids]))
		global_train = np.vstack(tuple([x_global[i] for i in ids]))
		# print('train articles shape = ' + str(x_train_articles.shape))
		# print('train headlines shape = ' + str(x_train_headlines.shape))
		# print('y train shape = ' + str(y_train.shape))

		x_valid_articles = x_articles[fold]
		x_valid_headlines = x_headlines[fold]
		y_valid = y_vals[fold]
		length_h_valid = lengths_h[fold]
		length_a_valid = lengths_a[fold]		
		global_valid = x_global[fold]

		fold_error = 0
		print('Training fold ' + str(fold))
		j = 0
		for epoch in range(5):
			
			# Training batches
			article_batches_train,headline_batches_train,output_batches_train,length_h_batches_train,length_a_batches_train, global_batches_train = create_batches(x_train_articles, 
			x_train_headlines, 
			y_train, 
			lengths_h_train, 
			lengths_a_train, 
			global_train)

			for i in range(len(article_batches_train)):
				# Training error
				epoch_error = model.session.run([model.error, model.train_fn], {
					model.inputs_articles: article_batches_train[i],
					model.inputs_headlines: headline_batches_train[i],
					model.outputs: output_batches_train[i],
					model.h_lengths: length_h_batches_train[i],
					model.a_lengths: length_a_batches_train[i],
					model.global_feats: global_batches_train[i]
				})[0]
				print('\tEpoch ' + str(j) + ' error = ' + str(epoch_error))				

				fold_error += epoch_error
				j += 1

		print('Training error (fold) = ' + str(fold_error / j) + '\n')
		
		# Validation batches
		article_batches_valid,headline_batches_valid,output_batches_valid,length_h_batches_valid,length_a_batches_valid, global_batches_valid = create_batches(x_valid_articles, 
		x_valid_headlines, 
		y_valid, 
		length_h_valid, 
		length_a_valid, 
		global_valid)
	
		all_pred_y_stances = []
		for i in range(len(article_batches_valid)):
			# cross-validation error
			pred_y_stances = model.session.run([model.pred_stance], {
					model.inputs_articles: article_batches_valid[i],
					model.inputs_headlines: headline_batches_valid[i],
					model.outputs: output_batches_valid[i],
					model.h_lengths: length_h_batches_valid[i],
					model.a_lengths: length_a_batches_valid[i],
					model.global_feats: global_batches_valid[i]
				})
			all_pred_y_stances = np.append(all_pred_y_stances, pred_y_stances)
		
		simple_y = np.array([array.tolist().index(1) for array in y_valid])
		'''
		f1_score = metrics.f1_score(simple_y, pred_y_stances, average='macro')
		print("F1 MEAN score: " + str(f1_score))
		f1_score_labels =  metrics.f1_score(simple_y, pred_y_stances, labels=[0, 1, 2, 3], average=None)
		print("F1 LABEL scores: " + str(f1_score_labels))
		'''
		# Convert to string labels for FNC scoring metric
		label_map = {0 : "agree", 1 : "disagree", 2 : "discuss", 3 : "unrelated"}
		simple_y_str = [label_map[label] for label in simple_y]
		pred_y_stances_str = [label_map[label] for label in all_pred_y_stances]
		report_score(simple_y_str, pred_y_stances_str)

	# assess performance on test set
	print('\n#### RUNNING ON HOLDOUT SET ####')

	# Test batches
	article_batches_test,headline_batches_test,output_batches_test,length_h_batches_test,length_a_batches_test,global_batches_test = create_batches(test_x_articles, 
	test_x_headlines, 
	test_y, 
	test_h_lengths, 
	test_a_lengths, 
	test_x_global)

	all_pred_y_test = []
	for i in range(len(article_batches_test)):
		pred_y_stances = model.session.run([model.pred_stance], {
				model.inputs_articles:  article_batches_test[i],
				model.inputs_headlines: headline_batches_test[i],
				model.outputs: output_batches_test[i],
				model.h_lengths: length_h_batches_test[i],
				model.a_lengths: length_a_batches_test[i],
				model.global_feats: global_batches_test[i]
			})
		all_pred_y_test = np.append(all_pred_y_test, pred_y_stances)

	simple_y = np.array([array.tolist().index(1) for array in test_y])
	f1_score = metrics.f1_score(simple_y, all_pred_y_test, average='macro')
	print("F1 MEAN score: " + str(f1_score))
	f1_score_labels =  metrics.f1_score(simple_y, all_pred_y_test, labels=[0, 1, 2, 3], average=None)
	print("F1 LABEL scores: " + str(f1_score_labels))

	# Convert to string labels for FNC scoring metric
	label_map = {0 : "agree", 1 : "disagree", 2 : "discuss", 3 : "unrelated"}
	simple_y_str = [label_map[label] for label in simple_y]
	pred_y_stances_str = [label_map[label] for label in all_pred_y_test]
	report_score(simple_y_str, pred_y_stances_str)

if __name__ == '__main__':
	main()

