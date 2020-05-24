'''
Plotting some 2-D Feature extraction plots.
Again shoutout to sci-kit learn
Sorry for all the single quotation marks btw, its not very pep-8 of me
'''


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import torch
import random
import matplotlib.pyplot as plt
from matplotlib import gridspec
import itertools
from mlxtend.plotting import plot_decision_regions
from sklearn.manifold import TSNE, LocallyLinearEmbedding

np.random.seed(420)
torch.manual_seed(420)
random.seed(420)

#File names:
data_file = '../data/datafinal.csv'

#Loading data:
data = pd.read_csv(data_file,index_col = None)


if __name__ == '__main__':

    #Make PCA model
    ''' Ok I don't think we are high dim enough for PCA '''
    pca = PCA(n_components = 2, svd_solver = 'auto')  #n<components < sqrt(157 = tot # of features)
    ica = FastICA(n_components = 2)
    lle = LocallyLinearEmbedding(n_components = 2)
    tsne = TSNE(n_components =2, verbose = 1, perplexity = 40, n_iter = 300)

    #Splitting Labels and features
    data_x = data.to_numpy()[:,1:]
   
    
    ''' For PCA: '''
    
    pca_x = pca.fit_transform(data_x)
    
    PCA_df = pd.DataFrame(data = pca_x , columns = ['PC1','PC2'])
    PCA_df = pd.concat([PCA_df,data.Winner],axis = 1)
    
    plt.figure(num = None, figsize = (8,8), dpi = 80, facecolor = 'w', edgecolor = 'k')

    classes = [1,0]
    colors = ['r','b']

    for cls, color in zip(classes, colors):
        plt.scatter(PCA_df.loc[PCA_df.Winner == cls, 'PC1'],
                    PCA_df.loc[PCA_df.Winner ==cls, 'PC2'],
                    c = color)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.title('2 Dimensional PCA')
    plt.legend(['Winner', 'Loser'])
    plt.show()
    
    ''' For ICA: '''

    ica_x = ica.fit_transform(data_x)
    
    ICA_df = pd.DataFrame(data = ica_x , columns = ['IC1','IC2'])
    ICA_df = pd.concat([ICA_df,data.Winner],axis = 1)
    
    plt.figure(num = None, figsize = (8,8), dpi = 80, facecolor = 'w', edgecolor = 'k')

    classes = [1,0]
    colors = ['r','b']

    for cls, color in zip(classes, colors):
        plt.scatter(ICA_df.loc[ICA_df.Winner == cls, 'IC1'],
                    ICA_df.loc[ICA_df.Winner ==cls, 'IC2'],
                    c = color)
    plt.xlabel('Independent Component 1')
    plt.ylabel('Independent Component 2')

    plt.title('2 Dimensional ICA')
    plt.legend(['Winner', 'Loser'])
    plt.show()

    ''' For LLE: '''

    lle_x = lle.fit_transform(data_x)
    
    LLE_df = pd.DataFrame(data = lle_x , columns = ['LLE1','LLE2'])
    LLE_df = pd.concat([LLE_df,data.Winner],axis = 1)
    
    plt.figure(num = None, figsize = (8,8), dpi = 80, facecolor = 'w', edgecolor = 'k')

    classes = [1,0]
    colors = ['r','b']

    for cls, color in zip(classes, colors):
        plt.scatter(LLE_df.loc[LLE_df.Winner == cls, 'LLE1'],
                    LLE_df.loc[LLE_df.Winner ==cls, 'LLE2'],
                    c = color)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.title('2 Dimensional LLE')
    plt.legend(['Winner', 'Loser'])
    plt.show()

    ''' For t-SNE: '''

    tsne_x = tsne.fit_transform(data_x)
    
    TSNE_df = pd.DataFrame(data = tsne_x , columns = ['TSNE1','TSNE2'])
    TSNE_df = pd.concat([TSNE_df,data.Winner],axis = 1)
    
    plt.figure(num = None, figsize = (8,8), dpi = 80, facecolor = 'w', edgecolor = 'k')

    classes = [1,0]
    colors = ['r','b']

    for cls, color in zip(classes, colors):
        plt.scatter(TSNE_df.loc[TSNE_df.Winner == cls, 'TSNE1'],
                    TSNE_df.loc[TSNE_df.Winner ==cls, 'TSNE2'],
                    c = color)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.title('2 Dimensional TSNE')
    plt.legend(['Winner', 'Loser'])
    plt.show()

    '''Showing the Overfit Decision Boundary
        Found a nice library B)
        '''

    forest = RandomForestClassifier(n_estimators = 100)

    
    forest.fit(pca_x, data.Winner)
    fig = plot_decision_regions(X=pca_x, y = data.Winner.to_numpy(), clf = forest, legend = 2)
    plt.title('Random Forest Boundary for PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
