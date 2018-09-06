import numpy as np
import pandas as pd
import json
import os
import string
from pprint import pprint
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import stem
import nltk

sno = stem.SnowballStemmer('english')
stop = stopwords.words('english')

print(os.getcwd())
filepath = os.getcwd() + "\\IntroDS\\week1\\"
filename = "Automotive_5.json"
filename_mod = "Automotive_5_mod.json"

def remove_stopWords(sw):
    sw = ' '.join(word for word in sw.split() if word not in stop)
    return sw

def stem_line(st):
    st = ' '.join(sno.stem(word) for word in st.split())
    return st

print(filepath + filename)
df = pd.read_json(filepath + filename, orient='records', lines='true')

df['reviewText'] = df.reviewText.apply(lambda x : str.lower(x))
df['reviewText'] = df.reviewText.apply(lambda x: remove_stopWords(x))
df['reviewText'] = df.reviewText.str.replace('[{}]'.format(string.punctuation), '')
df['reviewText'] = df.reviewText.apply(lambda x: stem_line(x))

df_pos = df.query('overall>3')
df_neg = df.query('overall<3')

np.savetxt(filepath+'pos.txt', df_pos.reviewText, fmt='%s')
np.savetxt(filepath+'neg.txt', df_neg.reviewText, fmt='%s')

# out = df.to_json(orient='records', lines='true')
# with open(filepath + filename_mod, 'w') as f:
#     f.write(out)