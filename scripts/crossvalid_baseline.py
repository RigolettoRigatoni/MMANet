'''
Cross Validation for the baseline
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
from sklearn.model_selection import cross_val_score

np.random.seed(420)
torch.manual_seed(420)
random.seed(420)

#File names:
data_file = '../data/crossvalid.csv'

data = pd.read_csv(data_file,index_col = None)
#print(data.shape)
#print(data.columns)


if __name__ == '__main__':

    #Splitting Labels and features
    x = data.to_numpy()[:,1:]
    y = data.to_numpy()[:,0]

    forest = RandomForestClassifier(n_estimators = 100, max_depth = None, min_samples_split = 2) 

    #Training the classifiers:
    scores = cross_val_score( forest, x, y, cv =5)
    print(scores.mean())
    print('^This is the accuracy')
    
