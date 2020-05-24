'''
Testing the baseline using feature extraction,
Done for 12 and 64 components when possible
'''


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE, LocallyLinearEmbedding
import torch
import random

np.random.seed(420)
torch.manual_seed(420)
random.seed(420)

#File names:
train_file = '../data/train.csv'
valid_file = '../data/valid.csv'
test_file = '../data/test.csv'

#Loading data:
train_data = pd.read_csv(train_file,index_col = None)
valid_data = pd.read_csv(valid_file, index_col = None)
test_data = pd.read_csv(test_file, index_col = None)
#print(data.shape)
#print(data.columns)


if __name__ == '__main__':

    #Make PCA model
    ''' Ok I don't think we are high dim enough for PCA '''
    pca_12 = PCA(n_components = 12, svd_solver = 'auto')  #n<components < sqrt(157 = tot # of features)
    pca_64 = PCA(n_components = 64, svd_solver = 'auto')

    ica_12 = FastICA( n_components = 12)
    ica_64 = FastICA( n_components = 64)

    lle_12 = LocallyLinearEmbedding(n_components = 12)
    lle_64 = LocallyLinearEmbedding(n_components = 64)

    tsne_2 = TSNE(n_components =2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_3 = TSNE(n_components = 3, verbose = 1, perplexity = 40, n_iter = 300)

    #Splitting Labels and features
    train_x = train_data.to_numpy()[:,1:]
    train_y = train_data.to_numpy()[:,0]

    valid_x = valid_data.to_numpy()[:,1:]
    valid_y = valid_data.to_numpy()[:,0]

    pca12_t_x = pca_12.fit_transform(train_x)
    pca12_v_x = pca_12.fit_transform(valid_x)

    pca64_t_x = pca_64.fit_transform(train_x)
    pca64_v_x = pca_64.fit_transform(valid_x)

    ica12_t_x = ica_12.fit_transform(train_x)
    ica12_v_x = ica_12.fit_transform(valid_x)

    ica64_t_x = ica_64.fit_transform(train_x)
    ica64_v_x = ica_64.fit_transform(valid_x)

    lle12_t_x = lle_12.fit_transform(train_x)
    lle12_v_x = lle_12.fit_transform(valid_x)

    lle64_t_x = lle_64.fit_transform(train_x)
    lle64_v_x = lle_64.fit_transform(valid_x)

    tsne2_t_x = tsne_2.fit_transform(train_x)
    tsne2_v_x = tsne_2.fit_transform(valid_x)

    tsne3_t_x = tsne_3.fit_transform(train_x)
    tsne3_v_x = tsne_3.fit_transform(valid_x)

    forest = RandomForestClassifier(n_estimators = 100, max_depth = None, min_samples_split = 2) 
    
    forest.fit(pca12_t_x,train_y)
    forest_pred = forest.predict(pca12_v_x)
    forest_accuracy = accuracy_score(valid_y, forest_pred)
    print('Validation Accuracy PCA 12:', forest_accuracy)
     
    forest.fit(pca64_t_x,train_y)
    forest_pred = forest.predict(pca64_v_x)
    forest_accuracy = accuracy_score(valid_y, forest_pred)
    print('Validation Accuracy PCA 64:', forest_accuracy)
      
    forest.fit(ica12_t_x,train_y)
    forest_pred = forest.predict(ica12_v_x)
    forest_accuracy = accuracy_score(valid_y, forest_pred)
    print('Validation Accuracy ICA 12:', forest_accuracy)
       
    forest.fit(ica64_t_x,train_y)
    forest_pred = forest.predict(ica64_v_x)
    forest_accuracy = accuracy_score(valid_y, forest_pred)
    print('Validation Accuracy ICA 64:', forest_accuracy)
     
    forest.fit(lle12_t_x,train_y)
    forest_pred = forest.predict(lle12_v_x)
    forest_accuracy = accuracy_score(valid_y, forest_pred)
    print('Validation Accuracy LLE 12:', forest_accuracy)
     
    forest.fit(lle64_t_x,train_y)
    forest_pred = forest.predict(lle64_v_x)
    forest_accuracy = accuracy_score(valid_y, forest_pred)
    print('Validation Accuracy LLE64:', forest_accuracy)
     
    forest.fit(tsne2_t_x,train_y)
    forest_pred = forest.predict(tsne2_v_x)
    forest_accuracy = accuracy_score(valid_y, forest_pred)
    print('Validation Accuracy TSNE 2:', forest_accuracy)
     
    forest.fit(tsne3_t_x,train_y)
    forest_pred = forest.predict(tsne3_v_x)
    forest_accuracy = accuracy_score(valid_y, forest_pred)
    print('Validation Accuracy TSNE 3:', forest_accuracy)
 
    
    forest.fit(train_x, train_y)
    forest_pred = forest.predict(valid_x)
    forest_acc = accuracy_score(valid_y, forest_pred)
    print('Validation Acceracy:' , forest_acc)
