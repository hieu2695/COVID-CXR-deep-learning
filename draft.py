#%% -------------------------------------- Import Lib --------------------------------------------------------------------
import torch
import torch.nn as nn
import os
import random
import numpy as np
from Helper import train_baseline_model, DataAug, learning_rate_finder, evaluation
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import torch.nn.functional as F
# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# load the data
x_valid, y_valid = np.load("train/x_valid.npy"), np.load("train/y_valid.npy")
x_test, y_test = np.load("train/x_test.npy"), np.load("train/y_test.npy")



#%% ------------------------------ DataLoader, Data Augmentation ----------------------------------------------------------
# convert to torch.Tensor
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

batch_test = 512

# apply transformation

valset = DataAug(x_valid, y_valid, transform = data_transform, length=len(x_valid))
testset = DataAug(x_test, y_test, transform = data_transform, length=len(x_test))

# generate DataLoader

valloader = DataLoader(valset, batch_size=batch_test)
testloader = DataLoader(testset, batch_size=batch_test)





n_classes = 3

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(3, 6, kernel_size=3),  # output (6x126x126)
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output (6x63x63)
            # Defining another 2D convolution layer
            nn.Conv2d(6, 16, kernel_size=3),  # output (16x61x61)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2), # output (16x30x30)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(16 * 30 * 30, n_classes),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

#%% --------------------------------- Preparation -----------------------------------------------------------------
model = Net()

path ="Model/baseline.pt"

#%% ----------------

# load best model weights
model.load_state_dict(torch.load(path))

TPR_val, FNR_val, score_val = evaluation(model, valloader)

TPR_test, FNR_test, score_test = evaluation(model, testloader)



#%%
print("Validation set: sensitivity = {:.4f}, specificity = {:.4f}, score = {:.4f}".format(TPR_val, FNR_val, score_val))
print("Testing set: sensitivity = {:.4f}, specificity = {:.4f}, score = {:.4f}".format(TPR_test, FNR_test, score_test))

