#%% -------------------------------------- Import Lib --------------------------------------------------------------------
import torch
import torch.nn as nn
import os
import random
import numpy as np
from Helper import train_baseline_model, DataAug, learning_rate_finder, evaluation, FocalLoss
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import torch.nn.functional as F



# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42

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

# one-hot encoding label
#y_train, y_valid, y_test = to_categorical(y_train, num_classes=n_classes), to_categorical(y_valid, num_classes=n_classes), to_categorical(y_test, num_classes=n_classes)

x_train, y_train = shuffle(x_train, y_train) ## shuffle training set

# check shape
#print(x_train.shape, y_train.shape)
#print(x_valid.shape, y_valid.shape)
#print(x_test.shape, y_test.shape)


#%% ------------------------------ DataLoader, Data Augmentation ----------------------------------------------------------
# convert to torch.Tensor
test_data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# batch size
batch_train = 128
batch_test = 512

# apply transformation
trainset = DataAug(x_train, y_train, transform = test_data_transform ,length=len(x_train))
valset = DataAug(x_valid, y_valid, transform = test_data_transform, length=len(x_valid))
testset = DataAug(x_test, y_test, transform = test_data_transform, length=len(x_test))

# generate DataLoader
trainloader = DataLoader(trainset, batch_size=batch_train)
valloader = DataLoader(valset, batch_size=batch_test)
testloader = DataLoader(testset, batch_size=batch_test)

# print loader size
#print(len(trainloader.sampler))
#print(len(valloader.sampler))
#print(len(testloader.sampler))



#%% ---------------------------------- Model Architecture ----------------------------------------------------------
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
LR = 3e-4
criterion = FocalLoss()
epochs = 1000
dir = os.path.dirname('Model/')
if not os.path.exists(dir):
    os.makedirs(dir)

path ="Model/baseline_focalloss.pt"
train_losses, val_losses = train_baseline_model(model, criterion, LR, epochs, "train_val",trainloader, valloader,  path)

#%% ----------------

# load best model weights
model.load_state_dict(torch.load(path))

TPR_val, FNR_val, score_val = evaluation(model, valloader)

TPR_test, FNR_test, score_test = evaluation(model, testloader)


#%% ---------------------------------------- Learning curve ------------------------------------------------------------
inds = np.arange(1,len(val_losses)+1)
plt.figure()
plt.plot(inds.astype(np.uint8), train_losses, label = "training loss")
plt.plot(inds.astype(np.uint8), val_losses, label = "validation loss")
plt.xlabel("Epoch")
plt.ylabel("Magnitude")
plt.title("Baseline model learning curve")
plt.legend(loc='best')
plt.xticks(np.arange(0, max(inds)+2, 3))
plt.show()

#%%
print("Validation set: sensitivity = {:.4f}, specificity = {:.4f}, score = {:.4f}".format(TPR_val, FNR_val, score_val))
print("Testing set: sensitivity = {:.4f}, specificity = {:.4f}, score = {:.4f}".format(TPR_test, FNR_test, score_test))
