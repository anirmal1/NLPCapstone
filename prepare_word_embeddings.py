import numpy as np

class WordEmbeddingsMap():

	def __init__(self):
		# CHANGE WORD EMBEDDING HERE
		f = open('/homes/iws/liliang/WindowsFolders/glove.6B.100d.txt')
		self.dataset = DataSet()
		self.word_vector_map = {}
		for line in f:
			split = line.split()
			word_key = split[0]	
			vector = np.array([float(x) for x in split[1:]])
			self.word_vector_map[word_key] = vector
	
		f.close()
		print("created word matrix")
		# return word_vector_map
	
	# should we return None for unk words?
	def get_embedding_for_word(self, word):
		return self.word_vector_map[word] if word in self.word_vector_map else None

	def get_embedding_for_text(self, text):
		

	def create_batches(self, n):
		pass

