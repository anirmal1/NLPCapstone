# methods to process word embeddings for fake news challenge

import gensim as gs
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords

stopwords = nltk.corpus.stopwords.words('english')

# builds and returns word2vec model using glove word vector embeddings
def build_word_vectors_model():
	model = KeyedVectors.load_word2vec_format('external_data/glove.6B.100d.txt', binary=False)
	return model

# get the average word vector for a sequence (like in the stanford paper, we can get the average for 
# the headline and then the average of each sentence in the article, pick the ~3 most similar article
# sentences and then get the distance between each sentence and the headline. 
# (see page 11 of stanford paper http://web.stanford.edu/class/cs224n/reports/2760496.pdf)
def get_avg_word_vector(model, seq):
	vector = np.zeros(100)
	count = 0
	for word in seq:
		if word in model.wv:
			vector += model.wv[word]
			count += 1

	average = vector / count
	return average
	
# returns the similarity between the two given sequences using the given model
# (IDEA: could use this by supplying claim and repeatedly supplying each (relevant) sentence of the article?)
def get_distance_between(model, seq1, seq2):
	global stopwords
	seq1_without_stop = [w for w in seq1 if w not in stopwords]
	seq2_without_stop = [w for w in seq2 if w not in stopwords]
	distance = model.wmdistance(seq1_without_stop, seq2_without_stop)
	return distance


