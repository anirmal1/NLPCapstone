import tensorflow as tf
from fnc_1_baseline_master.utils.score import LABELS
from word_embeddings import WordEmbeddings
import numpy as np

# Some constants
INPUT_SIZE = 100
OUTPUT_SIZE = 4
RNN_HIDDEN = 10
HIDDEN_OUTPUT_SIZE = 4
TINY = 1e-6
LEARNING_RATE = 0.01

model_path = 'lstm_model.ckpt'

def get_article_word_vectors(headline, article, stance, word_embeddings):
	article = article.split()[:200]
	article = word_embeddings.get_embedding_for_sentence(article)
	headline = word_embeddings.get_embedding_for_headline(headline) # trying to get equal length headline #get_embedding_for_sentence(h[i])
	stance_val = LABELS.index(stance)
	stance_arr = np.zeros(4)
	stance_arr[stance_val] = 1

	return np.array(headline), np.array(article), np.array(stance_arr)
 
def main():
	print('Starting test model...')
	session = tf.Session()
	# saver = tf.train.Saver(tf.all_variables())
	saver = tf.train.import_meta_graph(model_path + '.meta')
	saver.restore(session, save_path=model_path)
	print('Model restored.')
	
	# TODO complete this portion (feed in our own data, command line interface)

	embeddings = WordEmbeddings()
	headline = input('Headline? ')
	article = input('Article? ')
	true_label = input('True label? ')


	h, a, t, = get_article_word_vectors(headline, article, true_label, embeddings)
	
	pred_stances = session.run([pred_stance, train_fn], {
		inputs_articles: a,  # INSERT EMBEDDING
		inputs_headlines: h, # INSERT EMBEDDING
		outputs: t # INSERT EMBEDDING
	})[0]

	print(pred_stances)

if __name__ == '__main__':
	main()


