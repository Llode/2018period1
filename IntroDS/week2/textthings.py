#%%
import numpy as np
import pandas as pd
import os
from IPython.display import display, HTML
import matplotlib as mlp
import seaborn as sns
import simplejson
import json
from sklearn.feature_extraction.text import TfidfVectorizer

filepath = os.getcwd() + '\\IntroDS\\week1\\'

pos = pd.read_csv(filepath +'pos.txt')
neg = pd.read_csv((filepath +'neg.txt'))

def commonWords(file, amount):
    return pd.Series(' '.join(file).split()).value_counts()[:amount]
pos_words = commonWords(pos, 10)
neg_words = commonWords(neg, 10)
print(pos_words)
print(neg_words)

vectorizer = TfidfVectorizer()
response_pos = vectorizer.fit_transform(pos)
response_neg = vectorizer.fit_transform(neg)
print(response_pos)
print(response_neg)
