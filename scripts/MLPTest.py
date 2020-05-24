'''
Wrote a basic training loop for a simple MLP
'''

import pandas as pd
import torch
from time import time
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchsummary import summary
from torch.utils.data import DataLoader
from models import DeepNet, SimpleNet, FightDataset


def main(args):
    torch.manual_seed(args.seed)
    # data = pd.read_csv("../data/datafinal.csv")
    # label = data["Winner"].copy()
    # data = data.drop(columns=['Winner'])
    # data_np = data.to_numpy()
    # label_np = label.to_numpy()
    # x_train, x_test, y_train, y_test = train_test_split(data_np, label_np,
    #                                                     test_size=0.1)
    # Splitting and preparing the dataset and dataloader

    train_set = pd.read_csv("../data/train.csv", index_col=None)
    valid_set = pd.read_csv("../data/valid.csv", index_col=None)
    test_set = pd.read_csv("../data/test.csv", index_col=None)

    x_train = train_set.to_numpy()[:, 1:]
    y_train = train_set.to_numpy()[:, 0]
    x_valid = valid_set.to_numpy()[:, 1:]
    y_valid = valid_set.to_numpy()[:, 0]

    x_test = test_set.to_numpy()[:, 1:]
    y_test = test_set.to_numpy()[:, 0]

    train_data = FightDataset(x_train, y_train)
    valid_data = FightDataset(x_valid, y_valid)
    test_data = FightDataset(x_test, y_test)

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(valid_data, batch_size=len(x_valid), shuffle=True)
    test_loader = DataLoader(test_data, batch_size=len(x_test),
                             shuffle=True)

    model = DeepNet()
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=0.001)

    summary(model, (10, 157))
    # Initializing the list to hold accuracies and losses
    t_accuracystore = []
    v_accuracystore = []
    t_lossstore = []
    v_lossstore = []
    t = time()
    test_acc = 0
    for i in range(args.epochs):
        t_acc = 0
        model.train()
        for j, d in enumerate(train_loader, 0):

            inputs, label = d
            optimizer.zero_grad()
            predict = model(inputs.float())
            t_loss = loss_function(input=predict.squeeze(),
                                   target=label.float())
            t_loss.backward()
            optimizer.step()

            # Evaluating training accuracy
            for k in range(len(label)):
                predictsig = torch.sigmoid(predict)
                if round(predictsig[k].item()) == label[k]:
                    t_acc += 1

        v_acc = 0
        model.eval()
        # Evaluating validation accuracy
        for j, d in enumerate(val_loader, 0):
            inputs, label = d
            predict = model(inputs.float())
            v_loss = loss_function(input=predict.squeeze(),
                                   target=label.float())
            for k in range(len(label)):
                predictsig = torch.sigmoid(predict)
                if round(predictsig[k].item()) == label[k]:
                    v_acc += 1
        t_accuracystore.append(t_acc / len(train_data))
        v_accuracystore.append(v_acc / len(valid_data))
        t_lossstore.append(t_loss)
        v_lossstore.append(v_loss)
        print("Validation dataset accuracy: %5.3f" % (v_acc / len(valid_data)))

    for j, d in enumerate(test_loader, 0):
        inputs, label = d
        predict = model(inputs.float())
        test_loss = loss_function(input=predict.squeeze(),
                                  target=label.float())
        for k in range(len(label)):
            predictsig = torch.sigmoid(predict)
            if round(predictsig[k].item()) == label[k]:
                test_acc += 1

    elapsed = time() - t
    # print(elapsed)
    print("Test dataset accuracy: %5.3f" % (test_acc/len(test_data)))

    # Plotting accuracies for training and validation
    epoch_store = range(len(t_accuracystore))
    loss_store = range(len(t_lossstore))

    plt.plot(epoch_store, t_accuracystore, label='Train')
    plt.plot(epoch_store, v_accuracystore, label='Validation')
    plt.title("Accuracy over Batches")
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Batch #')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(loss_store, t_lossstore, label='Train')
    plt.plot(loss_store, v_lossstore, label='Validation')
    plt.title("Loss over Batches")
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Batch #')
    plt.ylabel('Accuracy')
    plt.show()
    torch.save(model, "model.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=420)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=27)
    args = parser.parse_args()

    main(args)
