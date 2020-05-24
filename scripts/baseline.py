'''
Code for training and evaluating baseline(s)
Did SVM, Logit, Random Forest and AdaBoost using decision trees
Oops got carried away
Shoutout to sci-kit learn
'''


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
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

'''
#3 round and 5 round seperated data:
train_file = '../data/five_round.csv'
valid_file = '../data/three_round.csv'
test_file = '../data/test.csv'
'''

#Loading data:
train_data = pd.read_csv(train_file,index_col = None)
valid_data = pd.read_csv(valid_file, index_col = None)
test_data = pd.read_csv(test_file, index_col = None)
#print(data.shape)
#print(data.columns)


if __name__ == '__main__':

    #Make PCA model
    ''' Ok I don't think we are high dim enough for PCA '''
    #pca = PCA(n_components = 12, svd_solver = 'auto')  #n<components < sqrt(157 = tot # of features)

    #Splitting Labels and features
    train_x = train_data.to_numpy()[:,1:]
    train_y = train_data.to_numpy()[:,0]

    valid_x = valid_data.to_numpy()[:,1:]
    valid_y = valid_data.to_numpy()[:,0]

    test_x = test_data.to_numpy()[:,1:]
    test_y = test_data.to_numpy()[:,0]

    '''
    train_x = pca.fit_transform(train_x)
    valid_x = pca.fit_transform(valid_x)
    test_x = pca.fit_transform(test_x)
    '''

    #Creating classifier objects:
    svm = SVC(gamma = 'auto')
    logit = LogisticRegression(solver = 'lbfgs')
    bigtree = DecisionTreeClassifier(max_depth = 1, splitter = 'random')
    forest = RandomForestClassifier(n_estimators = 100, max_depth = None, min_samples_split = 2) 
    ada = AdaBoostClassifier(base_estimator = None, n_estimators = 50, learning_rate = 1)
    neigh = KNeighborsClassifier(n_neighbors = 3)

    #Training the classifiers:
    svm.fit(train_x,train_y)
    logit.fit(train_x,train_y)
    forest.fit(train_x,train_y)
    ada.fit(train_x, train_y)
    neigh.fit(train_x,train_y)

    #Evaluating (Validation):
    #Do cross validation later?
    y_pred = svm.predict(valid_x)
    y_l_pred = logit.predict(valid_x)
    forest_pred = forest.predict(valid_x)
    ada_pred = ada.predict(valid_x)
    neigh_pred = neigh.predict(valid_x)
    
    v_accuracy = accuracy_score(valid_y,y_pred)
    logit_accuracy = accuracy_score(valid_y,y_l_pred)
    forest_accuracy = accuracy_score(valid_y, forest_pred)
    ada_accuracy = accuracy_score(valid_y, ada_pred)
    neigh_accuracy = accuracy_score(valid_y, neigh_pred)

    print('Validation Accuracy SVM:',v_accuracy)
    print('Validation Accuracy Logit',logit_accuracy)
    print('Validation Accuracy Forest:', forest_accuracy)
    print('Boosted Validation Accuracy:', ada_accuracy)
    print('Neighbors Validation Accuracy:', neigh_accuracy)
    
    
    #Evaluating (Testing):
    y_pred = svm.predict(test_x)
    y_l_pred = logit.predict(test_x)
    forest_pred = forest.predict(test_x)
    ada_pred = ada.predict(test_x)

    v_accuracy = accuracy_score(test_y,y_pred)
    logit_accuracy = accuracy_score(test_y,y_l_pred)
    forest_accuracy = accuracy_score(test_y, forest_pred)
    ada_accuracy = accuracy_score(test_y, ada_pred)

    print('Test Accuracy SVM:',v_accuracy)
    print('Test Accuracy Logit',logit_accuracy)
    print('Test Accuracy Forest:', forest_accuracy)
    print('Boosted Test Accuracy:', ada_accuracy)
    
    
