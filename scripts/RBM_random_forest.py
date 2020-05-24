'''
Code for training Random Forest baseline
Incorporates Restricted Boltzmann machine into training
Shoutout to Hinton for writing: "A Practical Guide to Training Restricted Boltzmann Machines"
I barely know what I'm doing
Will probably try VAEs or GANs later for generative but idk if that will help
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import accuracy_score
import torch
import random

np.random.seed(420)
torch.manual_seed(420)
random.seed(420)

#File names:
train_file = '../data/train.csv'
valid_file = '../data/valid.csv'
test_file = '../data/test.csv'

'''
Zero mean unit Variance does not work as well I do not recommend.
Makes sense since our values are all positive naturally and our labels are between 0 and 1
'''

'''
#Zero-mean data:
train_file = '../data/train_0mean.csv'
valid_file = '../data/valid_0mean.csv'
test_file = '../data/test_0mean.csv'
'''

#Loading data:
train_data = pd.read_csv(train_file,index_col = None)
valid_data = pd.read_csv(valid_file, index_col = None)
test_data = pd.read_csv(test_file, index_col = None)

if __name__ == '__main__':

    '''
    WIP: Incorporate RBM (this paper is so hard to read)
    Shoutout to mathematics for machine learning for teaching me
    '''

    #Splitting Labels and features
    train_x = train_data.to_numpy()[:,1:]
    train_y = train_data.to_numpy()[:,0]

    valid_x = valid_data.to_numpy()[:,1:]
    valid_y = valid_data.to_numpy()[:,0]

    test_x = test_data.to_numpy()[:,1:]
    test_y = test_data.to_numpy()[:,0]

    #Creating classifier objects:
    rf = RandomForestClassifier(n_estimators = 80, max_depth = None, min_samples_split = 2)
    rbm = BernoulliRBM(verbose = True)
    forest = Pipeline( steps = [('rbm', rbm), ('rf', rf)])

    #Training the classifiers:
    forest.fit(train_x,train_y)

    #Evaluating (Validation):
    #Do cross validation later?
    forest_pred = forest.predict(valid_x)
    
    forest_accuracy = accuracy_score(valid_y, forest_pred)
    
    print('Validation Accuracy Forest:', forest_accuracy)
    
    '''
    #Evaluating (Testing):
    forest_pred = forest.predict(test_x)
    
    forest_accuracy = accuracy_score(test_y, forest_pred)

    print('Test Accuracy Forest:', forest_accuracy)
    '''
    
