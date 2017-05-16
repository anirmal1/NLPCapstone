import tensorflow as tf

model_path = 'lstm_model.ckpt'

def main():
	print('Starting test model...')
	session = tf.Session()
	saver = tf.train.Saver()
	saver.restore(session, save_path=model_path)
	print('Model restored.')

if __name__ == '__main__':
	main()


