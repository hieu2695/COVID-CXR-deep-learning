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
from sklearn.model_selection import StratifiedKFold



# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 123

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# number of labels
n_classes = 3

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
# load the data
x_train, y_train = np.load("train/x_train.npy"), np.load("train/y_train.npy")
x_valid, y_valid = np.load("train/x_valid.npy"), np.load("train/y_valid.npy")
x_test, y_test = np.load("train/x_test.npy"), np.load("train/y_test.npy")

x_train = np.concatenate((x_train, x_valid))
y_train = np.concatenate((y_train, y_valid))

x_train, y_train = shuffle(x_train, y_train) ## shuffle training set

# check shape
#print(x_train.shape, y_train.shape)
#print(x_valid.shape, y_valid.shape)
#print(x_test.shape, y_test.shape)

# %% ---------------------------------- Model Architecture ----------------------------------------------------------
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
            nn.AvgPool2d(kernel_size=2, stride=2),  # output (16x30x30)
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

# %% ------------------------------ DataLoader, Data Augmentation ----------------------------------------------------------
# convert to torch.Tensor
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# batch size
batch_train = 128
batch_test = 512

LR = 3e-4
criterion = nn.CrossEntropyLoss()
epochs = 1000
dir = os.path.dirname('Model/')
if not os.path.exists(dir):
    os.makedirs(dir)

#%%
skf = StratifiedKFold(n_splits=4, random_state=SEED, shuffle=False)

validation_losses = []

for train_index, valid_index in skf.split(x_train, y_train):
    print("\nValidating:")
    data_train, data_valid = x_train[train_index], x_train[valid_index]
    label_train, label_valid = y_train[train_index], y_train[valid_index]



    # apply transformation
    trainset = DataAug(data_train, label_train, transform=data_transform, length=len(label_train))
    valset = DataAug(data_valid, label_valid, transform=data_transform, length=len(label_valid))

    # generate DataLoader
    trainloader = DataLoader(trainset, batch_size=batch_train)
    valloader = DataLoader(valset, batch_size=batch_test)

    model = Net()
    path ="baseline_CV.pt"
    train_losses, val_losses = train_baseline_model(model, criterion, LR, epochs, "train_val", trainloader, valloader,
                                                    path)
    validation_losses.append(val_losses)

#%% ----------- plot
fold_1 = min(validation_losses[0])
fold_2 = min(validation_losses[1])
fold_3 = min(validation_losses[2])
fold_4 = min(validation_losses[3])

average = (fold_1 + fold_2 + fold_3 + fold_4)/4

print("Fold 1: {:.4f}".format(fold_1))
print("Fold 2: {:.4f}".format(fold_2))
print("Fold 3: {:.4f}".format(fold_3))
print("Fold 4: {:.4f}".format(fold_4))
print("Average: {:.4f}".format(average))


