#%%
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import imageio
import numpy.random as npr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
from collections import Counter
import scipy

npr.seed(3456456)

filepath = os.getcwd() + "\\IntroDS\\week3\\HASYv2\\"

hasy_full = pd.read_csv(filepath+'hasy-data-labels.csv')
hasy = hasy_full.query('70<=symbol_id<=80')

#print(hasy.head(10))

pics = []

for pic in hasy.path:
    #print(pic.replace('/','\\'))
   pics.append(imageio.imread(filepath+pic.replace('/', '\\')))

labels = hasy.symbol_id

pics = np.asarray(pics)
pics2 = pics[:,:,:,0].reshape(len(pics), 32*32)

#print(labels)

pics2_train, pics2_test, labels_train, labels_test = train_test_split(pics2, labels, test_size=0.2)
print(pics2_train.shape)
print(labels_train.shape)

logreg = LogisticRegression()
logreg.fit(pics2_train, labels_train)

y_pred = logreg.predict(pics2_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(pics2_test, labels_test)))

def most_common(lst):
    data = Counter(lst)
    return max(lst, key=data.get)

naive = labels_train.values

right = 0
for i in labels_test:
    val = most_common(naive)
    ind = np.where(naive == val)
    #print(ind[0][0])
    naive = np.delete(naive, ind[0][0], axis=0)
    if i == val:
        right += 1

print('Accuracy of naive guesses on the test set: ', right/len(labels_test))

confutse = metrics.confusion_matrix(labels_test, y_pred)
print(confutse)

m = confutse.shape[0]
strided = np.lib.stride_tricks.as_strided
s0,s1 = confutse.strides
out = strided(confutse.ravel()[1:], shape=(m-1,m), strides=(s0+s1,s1)).reshape(m,-1)
print(out)