import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
from word_embeddings import WordEmbeddings
from sklearn import metrics
from fnc_1_baseline_master.utils.dataset import DataSet
from fnc_1_baseline_master.utils.generate_test_splits import kfold_split, get_stances_for_folds
from fnc_1_baseline_master.feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats, word_overlap_features
from fnc_1_baseline_master.utils.score import report_score, LABELS, score_submission
from fnc_1_baseline_master.utils.system import parse_params, check_version

model_path = 'lstm_model.ckpt' # for saving the model later

INPUT_SIZE = 100 # length of GLoVe word embeddings
RNN_HIDDEN = 10
OUTPUT_SIZE = 4
HIDDEN_OUTPUT_SIZE = 4
TINY = 1e-6
LEARNING_RATE = 0.0001

class Classifier(object):
	def __init__(self):
		self.session = tf.Session()

		# input/output placeholders
		self.inputs_articles = tf.placeholder(tf.float32, (None, 200, INPUT_SIZE), name='input_articles')
		self.inputs_headlines = tf.placeholder(tf.float32, (None, 30, INPUT_SIZE), name='inputs_headlines')	
		self.outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE), name='outputs') # TODO change to two dimensions

		# LSTM cells, TODO make these bidrectional!
		with tf.variable_scope('scope1') as scope1:  
			# Create cell
			self.cell_articles = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
			self.cell_articles = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(self.cell_articles, input_keep_prob=0.7, output_keep_prob=0.2)
			# Initialize batch size, initial states
			batch_size_articles= tf.shape(self.inputs_articles)[0]
			initial_state_articles = self.cell_articles.zero_state(batch_size_articles, tf.float32)
			# Hidden states, outputs
			self.rnn_outputs_articles, self.rnn_states_articles = tf.nn.dynamic_rnn(self.cell_articles, self.inputs_articles, initial_state=initial_state_articles, time_major=False)
		with tf.variable_scope('scope1') as scope1:
			scope1.reuse_variables() 
			# Create cell
			self.cell_headlines = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True, reuse=True)
			self.cell_headlines = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(self.cell_headlines, input_keep_prob=0.7, output_keep_prob=0.2) 
			# Initialize batch size, initial states
			batch_size_headlines= tf.shape(self.inputs_headlines)[0]
			initial_state_headlines = self.rnn_states_articles 
			# Hidden states, outputs
			self.rnn_outputs_headlines, self.rnn_states_headlines = tf.nn.dynamic_rnn(self.cell_headlines, self.inputs_headlines, initial_state=initial_state_headlines, time_major=False)

		# make prediction
		self.rnn_outputs = tf.concat([self.rnn_outputs_articles, self.rnn_outputs_headlines], 1)
		final_projection = lambda x: layers.linear(x, num_outputs=HIDDEN_OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)
		predicted_outputs = tf.map_fn(final_projection, self.rnn_outputs) # TODO project only for the final output!
		self.softmaxes = tf.nn.softmax(predicted_outputs[0:, -1, 0:])
		self.pred_stance = tf.argmax(self.softmaxes, 1)

		# cross entropy loss TODO compute cross entropy between softmax and expected output (a one-hot vector)
		self.error = -(self.outputs * tf.log(predicted_outputs + TINY) + (1.0 - self.outputs) * tf.log(1.0 - predicted_outputs + TINY))
		self.error = tf.reduce_mean(self.error)
		self.train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, name='train_fn').minimize(self.error)

		# accuracy TODO what is this even doing...
		self.accuracy = tf.reduce_mean(tf.cast(tf.abs(self.outputs - predicted_outputs) < 0.5, tf.float32))

###### END CLASSIFIER DEFINITION ######

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

		embeddings_list.append(article)
		headline_embeddings_list.append(headline)
		ys.append(y_val)

	return np.array(headline_embeddings_list), np.array(embeddings_list), np.array(ys, dtype=np.float32) # TODO return outputs in one-hot vector form (2 dimensions)
 
def main():
	# set up data
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

	for fold in fold_stances:
		x_headlines[fold], x_articles[fold], y_vals[fold] = get_articles_word_vectors(fold_stances[fold], d, embeddings)

	test_x_headlines, test_x_articles, test_y = get_articles_word_vectors(hold_out_stances, d, embeddings)
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
		print('train articles shape = ' + str(x_train_articles.shape))
		print('train headlines shape = ' + str(x_train_headlines.shape))
		print('y train shape = ' + str(y_train.shape))

		x_valid_articles = x_articles[fold]
		x_valid_headlines = x_headlines[fold]
		y_valid = y_vals[fold]
		
		fold_error = 0
		print('Training fold ' + str(fold))
		for epoch in range(10):
			
			batch_size = 512
			article_batches = []
			headline_batches = []
			output_batches = []

			start = 0
			while start < len(x_train_articles):
				article_chunk = x_train_articles[start:start + batch_size]
				headline_chunk = x_train_headlines[start:start + batch_size]
				output_chunk = y_train[start:start + batch_size]
				article_batches.append(article_chunk)
				headline_batches.append(headline_chunk)
				output_batches.append(output_chunk)
				start += batch_size
	
			for i in range(len(article_batches)):
				# Training error
				epoch_error = model.session.run([model.error, model.train_fn], {
					model.inputs_articles: article_batches[i],
					model.inputs_headlines: headline_batches[i],
					model.outputs: output_batches[i]
				})[0]
				print('\tEpoch error = ' + str(epoch_error))				

				fold_error += epoch_error

		print('Training error (fold) = ' + str(fold_error / 10.0) + '\n')
		
		# cross-validation error
		valid_accuracy, pred_y_stances = model.session.run([model.accuracy, model.pred_stance], {
				model.inputs_articles:  x_valid_articles,
				model.inputs_headlines: x_valid_headlines,
				model.outputs: y_valid
			})



	# assess performance on validation set
	print('\n#### RUNNING ON HOLDOUT SET ####')

	test_accuracy, pred_y_stances = model.session.run([model.accuracy, model.pred_stance], {
			model.inputs_articles:  test_x_articles,
			model.inputs_headlines: test_x_headlines,
			model.outputs: test_y
		})

	simple_y = np.array([array[0].tolist().index(1) for array in test_y])
	f1_score = metrics.f1_score(simple_y, pred_y_stances, average='macro')
	print("F1 MEAN score: " + str(f1_score))
	f1_score_labels =  metrics.f1_score(simple_y, pred_y_stances, labels=[0, 1, 2, 3], average=None)
	print("F1 LABEL scores: " + str(f1_score_labels))

	# Convert to string labels for FNC scoring metric
	label_map = {0 : "agree", 1 : "disagree", 2 : "discuss", 3 : "unrelated"}
	simple_y_str = [label_map[label] for label in simple_y]
	pred_y_stances_str = [label_map[label] for label in pred_y_stances]
	report_score(simple_y_str, pred_y_stances_str)

if __name__ == '__main__':
	main()

