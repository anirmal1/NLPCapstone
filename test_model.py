import tensorflow as tf
from fnc_1_baseline_master.utils.score import LABELS
from word_embeddings import WordEmbeddings
import numpy as np
import tensorflow.contrib.layers as layers
from fnc_1_baseline_master.feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats, word_overlap_features, discuss_features, get_sentiment_difference
import random
from fnc_lstm_class import Classifier

model_path = 'lstm_model.ckpt'

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
	X_vader_sentiment = gen_or_load_feats(get_sentiment_difference, h, b, "fnc_1_baseline_master/features/vader_sentiment."+name+".npy")
	#X_tfidf_headline, X_tfidf_bodies = gen_or_load_feats(get_tfidf, h, b, "fnc_1_baseline_master/features/tfidf."+name+".npy")
	
	# Pad X_vader_sentiment, X_tfidf_h, X_tfidf_b to fit feature matrix X
	vader_padding = np.zeros((len(h)-len(X_vader_sentiment), X_vader_sentiment.shape[1]))
	X_vader_sentiment = np.append(X_vader_sentiment, vader_padding, axis = 0)
	
	X = np.c_[X_vader_sentiment, X_discuss, X_hand, X_polarity, X_refuting, X_overlap]
	print(X)
	return X



def main():
	print('Starting test model...')

	model = Classifier()

	with model.graph.as_default():
		#session = tf.Session()
		with model.session as session:
			saver = tf.train.Saver(tf.all_variables())
			# model.session.run(tf.global_variables_initializer())
			# saver = tf.train.import_meta_graph(model_path + '.meta')
			saver.restore(session, save_path='/tmp/' + model_path)
			# model.session.run(tf.global_variables_initializer())
			print('Model restored.')

	
			embeddings = WordEmbeddings()
			
			while True:
				headline = input('Headline? ')
				article = input('Article? ')
				# dummy value
				true_label = 'discuss' #input('True label? ')

				h, a, t, l_h, l_a = get_articles_word_vectors(headline, article, true_label, embeddings)
				g_f = generate_features(headline, article)

				pred_stances = session.run([model.pred_stance], {
					model.inputs_articles: a,
					model.inputs_headlines: h,
					model.outputs: t, 
					model.h_lengths: l_h,
					model.a_lengths: l_a,
					model.global_feats: g_f
				})
				print('predicted label = ' + str(LABELS[pred_stances[0][0]]) + '\n')

if __name__ == '__main__':
	main()


