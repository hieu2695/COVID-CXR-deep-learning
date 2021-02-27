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
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc =  nn.Sequential(
    nn.Linear(num_ftrs, n_classes))

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

#%% --------------------------------- Preparation -----------------------------------------------------------------


path ="Model/resnet34_0.pt"

#%% ----------------

# load best model weights
model.load_state_dict(torch.load(path))

TPR_val, FNR_val, score_val = evaluation(model, valloader)

TPR_test, FNR_test, score_test = evaluation(model, testloader)



#%%
print("Validation set: sensitivity = {:.4f}, specificity = {:.4f}, score = {:.4f}".format(TPR_val, FNR_val, score_val))
print("Testing set: sensitivity = {:.4f}, specificity = {:.4f}, score = {:.4f}".format(TPR_test, FNR_test, score_test))

