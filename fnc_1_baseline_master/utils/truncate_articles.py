import pandas as pd
import numpy as np

def trunc_articles():
	df = pd.read_csv("../fnc-1/train_bodies.csv")
	articles = df["articleBody"]
	id_article_dict = {}
	for i, article in enumerate(articles):
		words = article.split()
		if len(words) > 200:
			words = words[:200]
		id_article_dict[df["Body ID"][i]] = words
	return id_article_dict
			
