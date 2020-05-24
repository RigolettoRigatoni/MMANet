'''
This script generates violin plots of class vs Winner for all classes
'''

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import torch
import random

np.random.seed(420)
torch.manual_seed(420)
random.seed(420)

#File names:
data_file = '../data/datafinal.csv'

#Load Data:
df = pd.read_csv(data_file, index_col = None)

if __name__ == '__main__':
    #Set some styles:
    sns.set(style = 'dark')
    #B is
    for col_name in df.columns:
        ax = sns.violinplot( x = 'Winner', y = col_name, data = df, showfliers = False)
        plt.show()
    
