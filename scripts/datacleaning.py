"""
Code for data cleaning
Balanced Red and Blue Wins
"""

import numpy as np
import pandas as pd
from models import *

np.random.seed(420)
data = pd.read_csv("../data/preprocessed_data.csv")
data = data.drop(columns=["B_Stance_Sideways", "title_bout"])
data_red = data[data["Winner"] == "Red"].copy()
data_blue = data[data["Winner"] == "Blue"].copy()

# Reset indices to randomly sample by index
data_blue = data_blue.reset_index(drop=True)
data_red = data_red.reset_index(drop=True)
indices = list(range(len(data_red)))
np.random.shuffle(indices)
data_red_final = data_red[data_red.index.isin(indices[:1796])].copy()

data_red_to_blue = data_red[data_red.index.isin(indices[1796:])].copy()
data_red_to_blue.loc[:, "Winner"] = "Blue"

# Segmenting the columns to concatenate in a different order
win_data = data_red_to_blue.iloc[:, 0:2]
blue_stats = data_red_to_blue.iloc[:, 2:68]
red_stats = data_red_to_blue.iloc[:, 68:134]
blue_age = data_red_to_blue.iloc[:, 134:135]
red_age = data_red_to_blue.iloc[:, 135:136]
weights = data_red_to_blue.iloc[:, 136:150]
blue_stance = data_red_to_blue.iloc[:, 150:154]
red_stance = data_red_to_blue.iloc[:, 154:158]

data_red_to_blue_final = pd.concat([win_data, blue_stats, red_stats, blue_age,
                                    red_age, weights, blue_stance, red_stance],
                                   axis=1)

# Replacing the switched columns with the original column names
data_red_to_blue_final.columns = list(data_red_final.columns)
data_final = pd.concat([data_blue, data_red_to_blue_final, data_red_final],
                       axis=0)
# Change all occurrences of "Red" to 1 and "Blue" as 0
data_final.loc[:, "Winner"] = (data_final["Winner"] == "Red").astype(int)
data_final = data_final.sample(frac=1, random_state=420).reset_index(drop=True)
data_final.to_csv("data_final.csv", index=False)
