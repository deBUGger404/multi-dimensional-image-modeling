import numpy as np
import pandas as pd

from models import *
from model_training import *
from utils.utils import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler


# variable init
batch_size = 32
val_split = 0.3
shuffle_dataset = True
random_seed = 123
visualization = True
num_epochs = 4
valid_loss_min = np.Inf
model_name = 'resnet'
num_classes = 2
input_dim = 8


# random image dataset for 8 dimension image
class random_dataset(Dataset):
    def __init__(self, input_dim):
        self.image = np.random.rand(5000,224,224,input_dim)
        self.labels = np.random.choice([0, 1], size=(5000,), p=[0.6,0.4])

    def __getitem__(self, index):
        image_1 = self.image[index]
        labels_1 = self.labels[index]
        image_2 = np.transpose(image_1, (2,1,0))
        return image_2, labels_1
    
    def __len__(self):
        return len(self.labels)



# Creating training and validation dataset
dataset = random_dataset(input_dim)
data_size = len(dataset)

indices = list(range(data_size))
split = int(np.floor(val_split * data_size))

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Create train ad val data loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            sampler=train_sampler,
                                            num_workers=8,
                                            pin_memory=True)
val_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            sampler=valid_sampler,
                                            num_workers=8,
                                            pin_memory=True)

# print('11')
# resnet, vgg, densenet, alexnet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft, input_size = initialize_model(model_name, num_classes, input_dim, use_pretrained=True)

# Put the model on the device:
model_ft = model_ft.to(device)

# define loss function
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.0001)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)

## model training
## model_train function save model automatically as model_multi_dim.pth
model_train(model_ft, optimizer, criterion, lr_scheduler, device, train_loader, val_loader, n_epochs=num_epochs).run()
