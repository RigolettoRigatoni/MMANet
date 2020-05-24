'''
Script for splitting the data based on number of rounds
Saved as data_3round.csv and data_5round.csv
Datasets are rebalanced and renormalized
'''

'''
NOTE: Rebalancing assumes more losers than winners
Worked for the dataset and I got lazy
Will need to change after more data is added
'''

import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
import torch

np.random.seed(420)
torch.manual_seed(420)
random.seed(420)
min_max_scaler = preprocessing.MinMaxScaler()

#File names:
data_file = '../data/datafinal.csv'

#Load into dataframe:
data = pd.read_csv(data_file, index_col = None)
col = data.columns

#Split into two dfs
five_df = data[data.no_of_rounds == 5].sort_values('Winner').reset_index(drop = True)
three_df = data[data.no_of_rounds == 3].sort_values('Winner')
three_df = three_df.iloc[1:,:].reset_index(drop = True)

'''
five_df.to_csv('../data/unb_5round.csv')
three_df.to_csv('../data/unb_3round.csv')
'''

#Rebalance:
five_count = five_df.Winner.value_counts()
three_count = three_df.Winner.value_counts()

five_to_swap = int((five_count[0] - five_count[1])/2)
three_to_swap = int((three_count[0] - three_count[1])/2)

for i in range(five_to_swap):
    for c in col:
        if c[0]=='B':
            c_red = 'R' + c[1:]

            red_val = five_df.loc[i,c_red]
            blue_val = five_df.loc[i,c]

            five_df.loc[i,c_red] = blue_val
            five_df.loc[i,c] = red_val

    five_df.loc[i,'Winner'] = 1

for i in range(three_to_swap):
    for c in col:
        if c[0]=='B':
            c_red = 'R' + c[1:]

            red_val = three_df.loc[i,c_red]
            blue_val = three_df.loc[i,c]

            three_df.loc[i,c_red] = blue_val
            three_df.loc[i,c] = red_val

    three_df.loc[i,'Winner'] = 1

#Normalize and shuffle:
five_df = five_df.sample(frac=1).reset_index(drop = True)
five_df = pd.DataFrame(min_max_scaler.fit_transform(five_df), columns = five_df.columns, index = five_df.index)

three_df = three_df.sample(frac=1).reset_index(drop = True)
three_df = pd.DataFrame(min_max_scaler.fit_transform(three_df), columns = three_df.columns, index = three_df.index)

#Save
five_df.to_csv('../data/five_round.csv', index = False)
three_df.to_csv('../data/three_round.csv', index = False)
