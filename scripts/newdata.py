"""
Code for data cleaning
Balanced Red and Blue Wins
"""

import numpy as np
import pandas as pd
from models import *

np.random.seed(420)
data = pd.read_csv("../data/data.csv")

data = data.fillna(0)

column_drop = ["B_HomeTown", "BStreak", "B_ID", "B_Location", "B_Name", "Date",
               "Event_ID", "Fight_ID", "Last_round", "R_HomeTown", "R_ID",
               "R_Location", "R_Name", "winby"]
data = data.drop(columns=column_drop)
data = data[data["winner"] != "no contest"]
data = data[data["winner"] != "draw"]

data_red = data[data["winner"] == "red"].copy()
data_blue = data[data["winner"] == "blue"].copy()

# Reset indices to randomly sample by index
data_blue = data_blue.reset_index(drop=True)
data_red = data_red.reset_index(drop=True)
indices = list(range(len(data_red)))
np.random.shuffle(indices)
data_red_final = data_red[data_red.index.isin(indices[:1139])].copy()

data_red_to_blue = data_red[data_red.index.isin(indices[1139:])].copy()
data_red_to_blue.loc[:, "winner"] = "blue"


blue_stats = data_red_to_blue.iloc[:, 0:4]
blue_fight_stats = data_red_to_blue.iloc[:, 4:439]
rounds = data_red_to_blue.iloc[:, 439:440]
red_stats = data_red_to_blue.iloc[:, 440:444]
red_fight_stats = data_red_to_blue.iloc[:, 444:879]
win = data_red_to_blue.iloc[:, 879:880]

data_red_to_blue_final = pd.concat([red_stats, red_fight_stats, rounds,
                                    blue_stats, blue_fight_stats, win], axis=1)

data_red_to_blue_final.columns = list(data_red_final.columns)
data_final = pd.concat([data_blue, data_red_to_blue_final, data_red_final],
                       axis=0)
for i in range(4, 439):
    data_final.iloc[:, i] = data_final.iloc[:, i] / data_final.iloc[:, 0]
for i in range(444, 879):
    data_final.iloc[:, i] = data_final.iloc[:, i] / data_final.iloc[:, 440]

data_final = data_final.fillna(0)
# Change all occurrences of "Red" to 1 and "Blue" as 0
data_final.loc[:, "winner"] = (data_final["winner"] == "red").astype(int)
data_final = data_final.sample(frac=1, random_state=420).reset_index(drop=True)
data_final.to_csv("../data/data_final_two.csv", index=False)

