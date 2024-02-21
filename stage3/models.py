import torch
from torch.utils.data import DataLoader
from torch import nn
from custom_datasets import MNISTDataSet, ORLDataSet, CIFARDataSet

# from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import recall_score, precision_score


class CNN_MNIST(nn.Module):

    def __init__(self):

        super(CNN_MNIST, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.layers(x)




class CNN_CIFAR(nn.Module):
    def __init__(self, setup=0) -> None:
        """
            setup = 0 default model \n
            setup = 1 default model with added padding, and dropout \n
            setup = 2 different moodel with additional CN, padding=1, kernel=3 and two dropouts \n
        
        """
        super(CNN_CIFAR, self).__init__()
        
        if setup == 0:
            self.labels = nn.Sequential(
                nn.Conv2d(3, 6, 5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(5*5*16, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        elif setup == 1:
            self.labels = nn.Sequential(
                nn.Conv2d(3, 6, 5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(6, 16, 5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(5*5*16, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        elif setup == 2:
            self.labels = nn.Sequential(
                nn.Conv2d(3, 33, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(33, 66, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(66, 132, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(4*4*132, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        elif setup == 3:
            self.labels = nn.Sequential(
                nn.Conv2d(3, 33, 3, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(2),
                nn.Conv2d(33, 66, 3, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(2),
                nn.Conv2d(66, 132, 3, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(2),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(4*4*132, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
            
            
        
        
    def forward(self, x):
        return self.labels(x)
        
class CNN_ORL(nn.Module):
    def __init__(self, setup=0) -> None:
        """
            setup = 0 default model \n
            setup = 1 same model, but with sigmoid activations \n
            setup = 2 default model with AvgPool instead of MaxPool \n
            setup = 3 default model with padding = 2 \n
            setup = 4 default model with padding = 3, and kernel = 5 \n
        """
        super(CNN_ORL, self).__init__()
        if setup == 0:
            self.labels = nn.Sequential(
                nn.Conv2d(1, 32, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(13824, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 40)    
            )
        elif setup == 1:
            self.labels = nn.Sequential(
                nn.Conv2d(1, 32, 3),
                nn.Sigmoid(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3),
                nn.Sigmoid(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3),
                nn.Sigmoid(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(13824, 256),
                nn.Sigmoid(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.Sigmoid(),
                nn.Linear(128, 40),
            )
        elif setup == 2:
            self.labels = nn.Sequential(
                nn.Conv2d(1, 32, 3),
                nn.ReLU(),
                nn.AvgPool2d(2),
                nn.Conv2d(32, 64, 3),
                nn.ReLU(),
                nn.AvgPool2d(2),
                nn.Conv2d(64, 128, 3),
                nn.ReLU(),
                nn.AvgPool2d(2),
                nn.Flatten(),
                nn.Linear(13824, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 40),
            )
        elif setup == 3:
            self.labels = nn.Sequential(
                nn.Conv2d(1, 32, 5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(10240, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 40),
            )
        elif setup == 4:
            self.labels = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(24960, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 40),
            )
        else:
            self.labels = nn.Sequential(
                nn.Conv2d(1, 32, 5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(19712, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 40),
            )
            
        
    def forward(self, x):
        return self.labels(x)



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
    accuracy = 100 * correct / size
    recall /= num_of_batches / 100
    precision /= num_of_batches / 100
    f1 = 2 * precision * recall / (precision + recall)
    loss /= num_of_batches
    print(
        f"""
        Metrics for {data_loader.dataset.name} on {mode} data:
        -- Accuracy: {accuracy:>.2f} % 
        -- Recall: {recall:>.2f} %
        -- Precision: {precision:>.2f} % 
        -- F1: {f1:>.2f} %
        -- Loss: {loss:>.2f} 
        """
    )
    return correct


# # def main():


if __name__ == "__main__":


    train_data = ORLDataSet(train=True)
    test_data = ORLDataSet(train=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    print(f"{len(test_loader.dataset)=}")
    cnn = CNN_ORL()
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
