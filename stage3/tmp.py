import torch
from torch.utils.data import DataLoader
from torch import nn
from custom_datasets import MNISTDataSet, ORLDataSet, CIFARDataSet

# from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import recall_score, precision_score


class CNN(nn.Module):

    # def __init__(self):
    #     super().__init__()
    #     self.network = nn.Sequential(
    #         nn.Conv2d(3, 32, kernel_size=3, padding=1),
    #         nn.ReLU(),
    #         nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

    #         nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

    #         nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

    #         nn.Flatten(),
    #         nn.Linear(256*4*4, 1024),
    #         nn.ReLU(),
    #         nn.Linear(1024, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, 10))

    # def forward(self, x):
    #     return self.network(x)

    # def __init__(self):

    #     super(CNN, self).__init__()
    #     self.conv = nn.Sequential(
    #         nn.Conv2d(3, 6, 5),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2),
    #         nn.Conv2d(6, 16, 5),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2),
    #         nn.Flatten(),
    #         nn.Linear(16*5*5, 120),
    #         # nn.Sigmoid(),
    #         # nn.Linear(720, 256),
    #         # nn.Sigmoid(),
    #         # nn.Linear(256, 120),
    #         nn.ReLU(),
    #         nn.Linear(120, 84),
    #         nn.ReLU(),
    #         nn.Linear(84, 10),
    #         nn.LogSoftmax(1)
    #         # nn.Softmax(1)
    #     )

    # def forward(self, x):
    #     return self.conv(x)

    def __init__(self):

        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
            # nn.Softmax(1)
        )
        # self.name
        

    def forward(self, x):
        output = self.conv(x)
        # print(f"{output=}")
        # output = torch.flatten(output)
        return self.linear(output)


def train(data_loader, model, optimizer, loss_function):
    model.train()
    avg_loss = 0
    for batch, (X, y) in enumerate(data_loader):
        loss = loss_function(model(X), y)
        avg_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss /= len(data_loader)
    return avg_loss


def test(data_loader, model, loss_function, mode="Test"):
    model.eval()
    num_of_batches = len(data_loader)
    size = len(data_loader.dataset)
    loss, correct = 0, 0
    recall, precision = 0, 0
    with torch.no_grad():
        for _, (X, y) in enumerate(data_loader):
            p = model(X)
            loss += loss_function(p, y).item()
            correct += (p.argmax(1) == y.argmax(1)).type(torch.float32).sum().item()
            y_true = y.argmax(1)
            y_pred = p.argmax(1)
            recall += recall_score(
                y_true, y_pred, average="macro", labels=(range(10)), zero_division=True
            )
            precision += precision_score(
                y_true, y_pred, average="macro", zero_division=True
            )
    accuracy = 100*correct / size
    recall /= (num_of_batches/100)
    precision /= (num_of_batches/100)
    f1 = 2 * precision * recall / (precision + recall)
    loss /= num_of_batches
    print(f"""
        Metrics for {data_loader.dataset.name} on {mode} data:
        -- Accuracy: {accuracy:>.2f} % 
        -- Recall: {recall:>.2f} %
        -- Precision: {precision:>.2f} % 
        -- F1: {f1:>.2f} %
        -- Loss: {loss:>.2f} 
        """)
    return correct


# def main():


if __name__ == "__main__":

    # a = [1,2,3,4,5,6,7,8,9, 0]
    # b = [0, 1, 2, 3, 4, 5, 6,7,8,9]
    # print(f"{recall_score(a, b, average='macro')=}")

    train_data = MNISTDataSet(train=True)
    test_data = MNISTDataSet(train=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    print(f"{len(test_loader.dataset)=}")
    cnn = CNN()
    cnn.to("cpu")
    opt = torch.optim.Adam(cnn.parameters(), lr=0.001)
    lf = nn.CrossEntropyLoss()

    epochs = 5
    train_loss = []

    for i in range(epochs):
        print(f"Epoch {i+1} ----------")
        train_loss.append(train(train_loader, cnn, opt, lf))
    # plt.plot(train_loss)
    # plt.show()

    print(f"\n {test(test_loader, cnn, lf)=}")
    print(f"\n {test(train_loader, cnn, lf)}")
