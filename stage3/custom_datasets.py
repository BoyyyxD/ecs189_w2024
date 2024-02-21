import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import pickle
from matplotlib import pyplot as plt
import torchvision
import numpy as np


class MNISTDataSet(Dataset):

    def __init__(self, train=False) -> None:

        with open("MNIST", "rb") as file:
            data = pickle.load(file)
        self.data = data["train" if train else "test"]
        self.num_of_labels = 10
        self.name = "MNIST"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tmp = self.data[index]
        label = torch.tensor(tmp["label"])
        label = F.one_hot(label, self.num_of_labels).float()
        image = torch.tensor(tmp["image"]).float()
        image = torch.reshape(image, (1, 28, 28))
        return image, label


class ORLDataSet(Dataset):
    
    def __init__(self, train=False):
        with open("ORL", "rb") as file:
            data = pickle.load(file)
        self.data = data["train" if train else "test"]
        self.num_of_labels = 40
        self.name = "ORL"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        label = torch.tensor(item["label"]) - 1
        label = F.one_hot(label, self.num_of_labels).float()
        
        image = torch.tensor(item["image"])[:, :, 0].float()
        image = torch.reshape(image, (1, 112, 92))
        return image, label


class CIFARDataSet(Dataset):
    
    def __init__(self, train=False) -> None:
        with open("CIFAR", "rb") as file:
            data = pickle.load(file)
        self.data = data["train" if train else "test"]
        self.num_of_labels = 10
        self.name = "CIFAR"   
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        image = torch.tensor(item["image"]).float()
        label = F.one_hot(torch.tensor(item["label"]), self.num_of_labels).float()
        image = torch.reshape(image, (3, 32, 32))
        image/=256
        return image, label

# get some random training images


# if __name__ == "__main__":
    # s = ORLDataSet()
    # rand_indx = torch.randint(len(s), size=(1,)).item()
    # a = []
    # for j in s:
    #     i, l = j
    #     print(f"{l=}")
    #     a.append(l)
        
    # l = torch.tensor(l)
    # l = torch.unique(l)
    # print(f"{l=}")
    # i, l = s[rand_indx]
    # # i/=256
    # # show_example(i, l)
    # # print()
    # # plt.imshow(i)
    # plt.imshow(i)
    # plt.show()
    # print(f"{i.shape=}")
    
    
    # i.ndim()
