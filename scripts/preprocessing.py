'''
Script for Preprocessing,
Normalizes each column by minmax scaling as well as zero-mean unit variance,
Splits data 80/10/10 Training,Validation and Testing
'''

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import torch
import random
from sklearn import preprocessing
from sklearn.preprocessing import scale

np.random.seed(420)
torch.manual_seed(420)
random.seed(420)
min_max_scaler = preprocessing.MinMaxScaler()

#File names:
data_file = '../data/datafinal.csv'

#Loading Data
data = pd.read_csv(data_file,index_col = None)

#Shuffling and normalizing columns two ways
df = data.sample( frac = 1).reset_index(drop=True)
data = pd.DataFrame( min_max_scaler.fit_transform(df), columns = df.columns, index = df.index)

data.to_csv('../data/crossvalid.csv', index = False)
#Splitting the Data (80-10-10)
tot_n = data.shape[0]/2
win_df = data[data['Winner'] == 1]
lose_df = data[data['Winner'] == 0]

print(len(win_df),len(lose_df))

n_train = int(0.8*tot_n)
n_valid = int(0.1*tot_n+1)

train_df = pd.concat([win_df.head(n_train), lose_df.head(n_train)])

win_df = win_df.iloc[n_train:]
lose_df = lose_df.iloc[n_train:]

valid_df = pd.concat([win_df.head(n_valid), lose_df.head(n_valid)])

test_df = pd.concat([win_df.iloc[n_valid:],lose_df.iloc[n_valid:]])

train_df = train_df.sample(frac = 1)
valid_df = valid_df.sample(frac = 1)
test_df = test_df.sample(frac = 1)

print('For the Training set: \n', train_df['Winner'].value_counts())
print('For the Validation set: \n', valid_df['Winner'].value_counts())
print('For the Testing set: \n', test_df['Winner'].value_counts())

#Saving the datasets
train_df.to_csv('../data/train.csv',index = False)
valid_df.to_csv('../data/valid.csv', index = False)
test_df.to_csv('../data/test.csv', index = False)

'''
This is copy and paste to make the zero mean data
'''
#Scale it

data = pd.DataFrame(scale(df, axis = 0), columns = df.columns, index = df.index)
data = pd.concat([df.iloc[:,0],data.iloc[:,1:]],axis = 1)
print(data.shape)

#Splitting the Data (80-10-10)
tot_n = data.shape[0]/2
win_df = data[data['Winner'] == 1]
lose_df = data[data['Winner'] == 0]

print(len(win_df),len(lose_df))

n_train = int(0.8*tot_n)
n_valid = int(0.1*tot_n+1)

train_df = pd.concat([win_df.head(n_train), lose_df.head(n_train)])

win_df = win_df.iloc[n_train:]
lose_df = lose_df.iloc[n_train:]

valid_df = pd.concat([win_df.head(n_valid), lose_df.head(n_valid)])

test_df = pd.concat([win_df.iloc[n_valid:],lose_df.iloc[n_valid:]])

train_df = train_df.sample(frac = 1)
valid_df = valid_df.sample(frac = 1)
test_df = test_df.sample(frac = 1)

print('For the Training set: \n', train_df['Winner'].value_counts())
print('For the Validation set: \n', valid_df['Winner'].value_counts())
print('For the Testing set: \n', test_df['Winner'].value_counts())

#Saving the datasets
train_df.to_csv('../data/train_0mean.csv',index = False)
valid_df.to_csv('../data/valid_0mean.csv', index = False)
test_df.to_csv('../data/test_0mean.csv', index = False)
