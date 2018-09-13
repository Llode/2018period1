#%%
import numpy as np
import pandas as pd
import os
from IPython.display import display, HTML
import matplotlib as mlp
import seaborn as sns
import simplejson
import json

filepath = os.getcwd() + "\\IntroDS\\week2\\"

test = pd.read_csv(filepath + 'test.csv')
gender_submission = pd.read_csv(filepath + 'gender_submission.csv')
df = pd.read_csv(filepath + 'train.csv')


#Fix the frame
del df['Ticket']
del df['Name']
del df['PassengerId']
#print(list(df))
df= df.assign(Deck=(df['Cabin'].str[0]))
#print(list(df))
del df['Cabin']
#print(df.dtypes)

#mode the deck and categoricalize
df['Deck'].fillna(df['Deck'].mode()[0], inplace=True)
#print(df.Deck)

df.Embarked = df.Embarked.astype('category')
df.Deck = df.Deck.astype('category')
df.Sex = df.Sex.astype('category')

#print(df.dtypes)

cat_cols = df.select_dtypes(['category']).columns
df[cat_cols] = df[cat_cols].apply(lambda x : x.cat.codes)

meanAge = df.Age.mean()
meanFare = df.Fare.mean()
#print(meanAge)
df['Age'].fillna(meanAge, inplace=True)
df['Fare'].fillna(meanFare, inplace=True)

df.fillna(df.mode(axis=0), inplace=True)
#print(df.head(0))

df.to_csv(filepath + 'wrangled_data.csv')
df.to_json(filepath + 'wrangled_data.csv')

""" with open(filepath + 'wrangled_data.json', 'r') as handle:
    parsed = json.load(handle)

with open(filepath + 'wrangled_data.json', 'wt') as out:
    res = json.dump(parsed, out, sort_keys=True, indent=4, separators=(',', ': '))
 """
print('Average traveller')
print(df[['Age', 'Fare']].median())
print(df.mode())

avg_alive = df.query('Survived==1')
avg_ded = df.query('Survived==0')

print('Average alive traveller')
print(avg_alive[['Age', 'Fare']].median())
print(avg_alive.mode())

print('quantiles')
print(avg_alive.quantile(0.5))

print('Average dead traveller')
print(avg_ded[['Age', 'Fare']].median())
print(avg_ded.mode())

print('quantiles')
print(avg_ded.quantile(0.5))

sns.set_palette("husl")
sns.pairplot(df, hue='Survived', kind='reg')
#sns.pairplot(avg_alive, hue='Survived', kind='reg')
#sns.pairplot(avg_ded, hue='Survived')
#pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
