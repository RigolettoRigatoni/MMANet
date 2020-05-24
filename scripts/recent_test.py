import pandas as pd
import numpy as np
import torch
from models import *

np.random.seed(420)
data = pd.read_csv("../data/test_fights_raw.csv")
data = data.drop(columns=data.columns[0:2])
data.loc[:, "Winner"] = (data["Winner"] == "Red").astype(int)
for i in range(1, 158):
    data.iloc[:, i] = data.iloc[:, i] / data.iloc[:, i].max()

data = data.fillna(0)
label = data["Winner"].copy()
data = data.drop(columns=['Winner'])

data_np = data.to_numpy()
label_np = label.to_numpy()

label = torch.tensor(label_np)
features = torch.tensor(data_np)
model = torch.load("model.pt")
model.eval()

predict = model(features.float())
predictsig = torch.sigmoid(predict)
acc = 0

for k in range(len(label)):
    if round(predictsig[k].item()) == label[k]:
        acc += 1

print(acc/len(label))
print(predictsig.round())
