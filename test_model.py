import tensorflow as tf
from fnc_1_baseline_master.utils.score import LABELS
from word_embeddings import WordEmbeddings
import numpy as np
import tensorflow.contrib.layers as layers
from fnc_1_baseline_master.feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats, word_overlap_features, discuss_features, get_sentiment_difference
import random

# Some constants
INPUT_SIZE = 100
OUTPUT_SIZE = 4
RNN_HIDDEN = 10
HIDDEN_OUTPUT_SIZE = 4
TINY = 1e-6
LEARNING_RATE = 0.01

model_path = 'lstm_model.ckpt'

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

		window = 4

		# LSTM cells, TODO make these bidrectional!
		with tf.variable_scope('scope1') as scope1:  
			# Create cell
			self.cell_articles_fw = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
			self.cell_articles_fw = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(self.cell_articles_fw, input_keep_prob=0.7, output_keep_prob=0.2)
			self.cell_articles_fw = tf.contrib.rnn.AttentionCellWrapper(self.cell_articles_fw, window, state_is_tuple=True)
			self.cell_articles_bw = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
			self.cell_articles_bw = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(self.cell_articles_bw, input_keep_prob=0.7, output_keep_prob=0.2)
			self.cell_articles_bw = tf.contrib.rnn.AttentionCellWrapper(self.cell_articles_bw, window, state_is_tuple=True)
			self.rnn_outputs_articles, self.rnn_states_articles = 	tf.nn.bidirectional_dynamic_rnn(self.cell_articles_fw, self.cell_articles_bw, self.inputs_articles, dtype=tf.float32)

		with tf.variable_scope('scope1') as scope1:
			scope1.reuse_variables() 
			# Create cell
			self.cell_headlines_fw = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True, reuse=True)
			self.cell_headlines_fw = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(self.cell_headlines_fw, input_keep_prob=0.7, output_keep_prob=0.2)
			self.cell_headlines_fw = tf.contrib.rnn.AttentionCellWrapper(self.cell_headlines_fw, window, state_is_tuple=True, reuse=True) 
			self.cell_headlines_bw = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True, reuse=True)
			self.cell_headlines_bw = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(self.cell_headlines_bw, input_keep_prob=0.7, output_keep_prob=0.2) 
			self.cell_headlines_bw = tf.contrib.rnn.AttentionCellWrapper(self.cell_headlines_bw, window, state_is_tuple=True, reuse=True)
			self.rnn_outputs_headlines, self.rnn_states_headlines = tf.nn.bidirectional_dynamic_rnn(self.cell_headlines_fw, self.cell_headlines_bw, self.inputs_headlines, dtype=tf.float32)

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






def get_articles_word_vectors(h_, b_, t, word_embeddings):
	y = []
	h = []
	b = []
	length_a, length_h = [],[]
	zeros = np.zeros(4)
	zeros[LABELS.index(t)] = 1
	y.append(np.array(zeros))
	h.append(h_)
	b_split = b_.split()[:200]
	b.append(b_split)
	min_length_a = min(200, len(b_)) - 1
	length_a.append(np.array([0, min_length_a]))
	min_length_h = min(30, len(h_)) - 1
	length_h.append(np.array([0, min_length_h]))

	length = len(b_split)
	for i in range(200 - length):
		b_split.append(' ')

	embeddings_list = []
	headline_embeddings_list = []
	ys = []
	for i in range(len(b)):
		article = word_embeddings.get_embedding_for_sentence(b[i])
		headline, length = word_embeddings.get_embedding_for_headline(h[i]) # trying to get equal length headline #get_embedding_for_sentence(h[i])
		y_val = y[i]

		embeddings_list.append(article)
		headline_embeddings_list.append(headline)
		ys.append(y_val)

	return np.array(headline_embeddings_list), np.array(embeddings_list), np.array(ys, dtype=np.float32), np.array(length_h), np.array(length_a)  # TODO return outputs in one-hot vector form (2 dimensions)



def generate_features(h, b):
	name = (str(b[:10]) + str(h[:10]))
	name = name.replace(' ', '')
	h = [h]
	b = [b]
	# Transform b to be one string instead of a list of words
	X_overlap = gen_or_load_feats(word_overlap_features, h, b, "fnc_1_baseline_master/features/overlap."+name+".npy")
	X_refuting = gen_or_load_feats(refuting_features, h, b, "fnc_1_baseline_master/features/refuting."+name+".npy")
	X_polarity = gen_or_load_feats(polarity_features, h, b, "fnc_1_baseline_master/features/polarity."+name+".npy")
	X_hand = gen_or_load_feats(hand_features, h, b, "fnc_1_baseline_master/features/hand."+name+".npy")
	X_discuss = gen_or_load_feats(discuss_features, h, b, "fnc_1_baseline_master/features/discuss."+name+".npy")
	# X_vader_sentiment = gen_feats(get_sentiment_difference, h, b, "fnc_1_baseline_master/features/vader_sentiment."+name+".npy")
	#X_tfidf_headline, X_tfidf_bodies = gen_feats(get_tfidf, h, b, "fnc_1_baseline_master/features/tfidf."+name+".npy")
	X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
	print(X)
	return X






def main():
	print('Starting test model...')
	session = tf.Session()
	# saver = tf.train.Saver(tf.all_variables())
	model = Classifier()
	saver = tf.train.import_meta_graph(model_path + '.meta')
	saver.restore(session, save_path=model_path)
	model.session.run(tf.global_variables_initializer())
	print('Model restored.')

	print(tf.all_variables())
	
	# TODO complete this portion (feed in our own data, command line interface)
	embeddings = WordEmbeddings()
	
	while True:
		headline = input('Headline? ')
		article = input('Article? ')
		true_label = input('True label? ')

		h, a, t, l_h, l_a = get_articles_word_vectors(headline, article, true_label, embeddings)
		g_f = generate_features(headline, article)

		pred_stances = model.session.run([model.pred_stance, model.train_fn], {
			model.inputs_articles: a,  # INSERT EMBEDDING
			model.inputs_headlines: h, # INSERT EMBEDDING
			model.outputs: t, # INSERT EMBEDDING
			model.h_lengths: l_h,
			model.a_lengths: l_a,
			model.global_feats: g_f
		})[0]

		print('predicted label = ' + str(LABELS[pred_stances[0]]) + '\n')

if __name__ == '__main__':
	main()


