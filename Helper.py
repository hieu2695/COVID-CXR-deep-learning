import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

#%%-------------------- Data Augmentation -----------------
class DataAug(Dataset):
    """
    learning from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    def __init__(self, data, targets, transform=None, length=None):

        self.transform = transform
        self.data = data.astype(np.uint8)
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.length = length

    def __getitem__(self, index):
        index = index % len(self.data)
        x = self.data[index]
        y = self.targets[index]

        if self.transform is not None:
            x = self.transform(x)

        return x , y

    def __len__(self):
        return self.length


#%%------------------ Training processs ---------------------------
def train_baseline_model(model, criterion, LR, epochs, mode, trainloader, testloader, path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    count = 0
    train_losses, val_losses = [], []
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Start training ...")

    for i in range(epochs):
        print("=" * 30)
        print('Epoch {}/{}'.format(i+1, epochs))
        print('-' * 30)

        train_loss = 0.0
        val_loss = 0.0


        # set model to training mode
        model.train()

        for data, label in trainloader:
            data = data.to(device)
            label = label.type(torch.LongTensor)
            label = label.to(device)

            # make gradients zero
            optimizer.zero_grad()
            # predictions
            train_logits = model(data)
            # loss function
            loss = criterion(train_logits, label)
            # backpropagation
            loss.backward()
            # parameter update
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        # track the loss for each epoch
        train_loss = train_loss/len(trainloader.sampler)
        train_losses.append(train_loss)

        if mode == "train":
            if i == 0:
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, path)
                best_loss = train_loss

            else:
                if  train_loss < best_loss:
                    best_loss = train_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, path)
                    count = 0
                else:
                    count = count + 1
            print('Epoch Loss: {:.6f}'.format(train_loss))

        else:

            # validation model
            model.eval()  # change to evaluate mode
            for data, label in testloader:
                data = data.to(device)
                label = label.type(torch.LongTensor)
                label = label.to(device)

                with torch.no_grad():
                    val_logits = model(data)
                    loss = criterion(val_logits, label)
                    val_loss += loss.item()*data.size(0)


            # track the loss
            val_loss = val_loss / len(testloader.sampler)
            val_losses.append(val_loss)


            # save model
            if i == 0:
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, path)
                best_loss = val_loss

            else:
                if  val_loss < best_loss:
                    best_loss = val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, path)
                    count = 0
                else:
                    count = count + 1
            print('Train Loss: {:.6f} |  Validation Loss: {:.6f}'.format(train_loss, val_loss))

        if mode == "train":
            if (count == 10) or (best_loss < 1e-3):
                break
        else:
            if count == 3:
                break


    print("=" * 20)
    print("Training Complete.")
    print("Best_loss: {:.6f}".format(best_loss))


    return train_losses, val_losses


#%%------------------------ Focal Loss -----------------------------
class FocalLoss(nn.Module):
    """
    base source code:
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    """

    def __init__(self, alpha=1, gamma=1.5, logits=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits


    def forward(self, inputs, targets):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.alpha = self.alpha.to(device)

        #inputs = inputs.view(-1)
        #targets = targets.view(-1)

        if self.logits:
            CE_loss = F.cross_entropy(inputs, targets, reduction="none")
        else:
            CE_loss = F.cross_entropy(inputs, targets, reduction="none")

        targets = targets.type(torch.long)
        at = self.alpha

        pt = torch.exp(-CE_loss)
        F_loss = at*(1-pt)**self.gamma * CE_loss

        return torch.mean(F_loss)

#%% -------------------------------------- Finding learning rate -------------------------------------------------------
def learning_rate_finder(model, criterion, min_lr, max_lr, iteration, trainloader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    factor = np.exp(np.log(max_lr/min_lr)/ iteration)
    LR = min_lr
    count = 0
    train_losses, rates = [], []

    print("Finding learning rate ...")

    while count < iteration:
        # set model to training mode
        model.train()

        for data, label in trainloader:
            count = count + 1
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            optimizer
            data = data.to(device)
            label = label.type(torch.LongTensor)
            label = label.to(device)

            # make gradients zero
            optimizer.zero_grad()
            # predictions
            train_logits = model(data)
            # loss function
            loss = criterion(train_logits, label)
            # get training loss
            train_loss = loss.item()
            # backpropagation
            loss.backward()
            # parameter update
            optimizer.step()


            # track the loss for each iteration
            train_losses.append(train_loss)

            # track the learning rate for each iteration
            rates.append(LR)

            # update learning rate
            LR = LR * factor

            if count == iteration:
                break

    plt.figure()
    plt.plot(rates, train_losses)
    plt.gca().set_xscale("log")
    plt.xlabel("learning rate (log scale)")
    plt.ylabel("loss")
    plt.show()

    return train_losses, rates

#%%---
def evaluation(model, loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    running_preds = []
    running_targets = []
    model.eval()  # change to evaluate mode
    for data, label in loader:
        data = data.to(device)
        label = label.to(device)
        label = label.type(torch.LongTensor)

        true_targets = label.cpu().detach().numpy()
        for true_target in true_targets:
            running_targets.append(true_target)

        with torch.no_grad():
            test_logits = model(data)

        _, predicted = torch.max(test_logits, 1)
        predicted = predicted.cpu().detach().numpy()
        for pred in predicted:
            running_preds.append(pred)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(running_targets)):
        if running_targets[i] == 0:
            if running_preds[i] == 0:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            if running_preds[i] == 0:
                FP = FP + 1
            else:
                TN = TN + 1

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    score = (sensitivity + specificity) / 2

    return sensitivity, specificity, score
