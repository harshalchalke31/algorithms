import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt


class ViTDataset():
    def __init__(self, BATCH_SIZE, IMG_SIZE):
        self.batch_size= BATCH_SIZE
        self.img_size = IMG_SIZE
        self.transform_train = transforms.Compose([
                            transforms.Resize(self.img_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(self.img_size,pad_if_needed=4),
                            transforms.RandAugment(num_ops=2, magnitude=12), # best - 9
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.505,0.456,0.489],
                                                 std=[0.229,0.224,0.225])
                        ])
        self.transform_test = transforms.Compose([
                            transforms.Resize(self.img_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.505,0.456,0.489],
                                                 std=[0.229,0.224,0.225])
                        ])
    def train_loader(self):
        train_data = datasets.CIFAR100(root='./data/',
                                       train=True,
                                       transform=self.transform_train,
                                       download=True)
        return DataLoader(dataset=train_data,
                          batch_size=self.batch_size,
                          shuffle=True)
    def test_loader(self):
        test_data = datasets.CIFAR100(root='../data/',
                                       train=False,
                                       transform=self.transform_test,
                                       download=True)
        return DataLoader(dataset=test_data,
                          batch_size=self.batch_size,
                          shuffle=False)
    
if __name__=="__main__":
    data = ViTDataset()
    train_data  = data.train_loader()
    print(len(train_data))