import numpy as np

class WordEmbeddings():

	def __init__(self):
		# Load GloVe word embeddings
		f = open('/homes/iws/liliang/WindowsFolders/glove.6B.100d.txt')
		self.word_vector_map = {}
		for line in f:
			split = line.split()
			word_key = split[0]	
			vector = np.array([float(x) for x in split[1:]])
			self.word_vector_map[word_key] = vector
	
		f.close()
		print("Loaded GloVe word embeddings")
	
	# should we return None for unk words?
	def get_embedding_for_word(self, word):
		return self.word_vector_map[word] if word in self.word_vector_map else None

	def get_embedding_for_sentence(self, sentence):
		vectorList = []
		for word in sentence:
			if self.get_embedding_for_word(word) is not None:
				vectorList.append(self.word_vector_map[word])
		return np.hstack(vectorList)

	def create_batches(self, n):
		pass

