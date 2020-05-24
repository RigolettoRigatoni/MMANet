'''
Wrote a basic training loop for a simple MLP
'''
import pandas as pd
import torch
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from models import *
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_score


def main(args):
    torch.manual_seed(args.seed)
    data = pd.read_csv("../data/datafinal.csv")
    label = data["Winner"].copy()
    data = data.drop(columns=['Winner'])
    data_np = data.to_numpy(dtype=np.float32)
    label_np = label.to_numpy(dtype=np.int64)
    x_train, x_test, y_train, y_test = train_test_split(data_np, label_np,
                                                        test_size=0.1)

    model = DeepNetCross()
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr)

    net = NeuralNetClassifier(model, #criterion=torch.nn.BCELoss(),
                              max_epochs=100, # optimizer=torch.optim.Adam,
                              lr=1e-2)
    scores = cross_val_score(net, data_np, label_np, cv=3, scoring="accuracy")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=420)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    main(args)
