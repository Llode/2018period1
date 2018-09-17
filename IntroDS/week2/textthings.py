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
from collections import Counter
from sklearn.decomposition import PCA

filepath = os.getcwd() + '\\IntroDS\\week1\\'


def dummy_fun(doc):
    return doc

tokenize = lambda doc: doc.lower().split(" ")

pos = pd.read_csv(filepath +'pos.txt')
neg = pd.read_csv(filepath +'neg.txt')

def commonWords(file, amount):
    return pd.Series(' '.join(file).split()).value_counts()[:amount]

with open(filepath +'neg.txt') as input_file:
    #build a counter from each word in the file
    count = Counter(word for line in input_file
                         for word in line.split())

neg_words = count.most_common(10)

with open(filepath +'pos.txt') as input_file:
    #build a counter from each word in the file
    count = Counter(word for line in input_file
                         for word in line.split())


pos_words = count.most_common(10)


print('pos top10 \n', pos_words)
print('neg top10 \n', neg_words)

vectorizerX = TfidfVectorizer()
X = vectorizerX.fit_transform(pos)
print(X.shape)

pca = PCA(n_components=2).fit(X.toarray())
data2D = pca.transform(X)
plt.scatter(data2D[:,0], data2D[:,1], c=data.target)
plt.show()

Y = vectorizerY.fit_transform(neg)
fitted_neg = vectorizerY.fit(neg)

print('positive vocab: ', vectorizerX.vocabulary_)
print('negative vocab: ', vectorizerY.vocabulary_)