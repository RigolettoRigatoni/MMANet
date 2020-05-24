import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
import torch.utils.data as data


class FightDataset(data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return [self.X[index, :], self.y[index]]


class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.fc1 = nn.Linear(157, 10)
        self.fc1_BN = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(10, 10)
        self.fc2_BN = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10, 10)
        self.fc3_BN = nn.BatchNorm1d(10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = sigmoid(self.fc1_BN(self.fc1(x)))
        x = sigmoid(self.fc2_BN(self.fc2(x)))
        x = sigmoid(self.fc3_BN(self.fc3(x)))
        x = self.fc4(x)
        return x


class DeepNet2(nn.Module):
    def __init__(self):
        super(DeepNet2, self).__init__()
        self.fc1 = nn.Linear(879, 100)
        self.fc1_BN = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 24)
        self.fc2_BN = nn.BatchNorm1d(24)
        self.fc3 = nn.Linear(24, 1)

    def forward(self, x):
        x = sigmoid(self.fc1_BN(self.fc1(x)))
        x = sigmoid(self.fc2_BN(self.fc2(x)))
        x = sigmoid(self.fc3(x))
        return x

class DeepNetCross(nn.Module):
    def __init__(self):
        super(DeepNetCross, self).__init__()
        self.fc1 = nn.Linear(157, 10)
        self.fc1_BN = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(10, 10)
        self.fc2_BN = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10, 10)
        self.fc3_BN = nn.BatchNorm1d(10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = sigmoid(self.fc1_BN(self.fc1(x)))
        x = sigmoid(self.fc2_BN(self.fc2(x)))
        x = sigmoid(self.fc3_BN(self.fc3(x)))
        x = sigmoid(self.fc4(x))
        return x

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(879, 10)
        self.fc1_BN = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = sigmoid(self.fc1_BN(self.fc1(x)))
        x = sigmoid(self.fc2(x))
        return x
