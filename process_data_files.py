# some quick methods for processing the csv

import pandas as pd
import numpy as np

# returns a dictionary from body id -> article body (might need to do something different depending on how the split method works 
def get_article_body_ids():
	df = pandas.read_csv('fnc-1-baseline-master/fnc-1/train_bodies.csv')
	dictionary = df.set_index('Body ID')['articleBody'].to_dict()
	return dictionary

# returns a dictionary from (headline, article id) -> stance
def get_claim_article_pairs():
	df = pandas.read_csv('fnc-1-baseline-master/fnc-1/train_stances.csv')
	claim_articles = {}
	for row in df.rows:
		claim = df['Headline']
		article_id = df['Body ID']
		stance = df['Stance']
		claim_articles[(claim, article)] = stance

	return claim_articles

