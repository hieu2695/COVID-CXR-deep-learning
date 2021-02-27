import numpy as np
import matplotlib.pyplot as plt
import os
import random
#%% -------------------------------------- Load data -------------------------------------------------------------------
x = np.load("COVID_npy/input.npy")
y = np.load("COVID_npy/target.npy")

#%% -------------------------------------- Class distribution ----------------------------------------------------------
n_classes = 3
value, count = np.unique(y, return_counts = True)

inds = np.arange(n_classes)
labels = ["COVID","NORMAL","NON-COVID PNEUMONIA"]

plt.figure()
for i in range(n_classes):
    plt.bar(inds[i], count[i], width=0.5, label = labels[i])
    plt.text(inds[i] - 0.075, count[i]+ 100, s = str(count[i]))
plt.legend(loc="best")
plt.xticks(inds, labels, rotation = 0)
plt.title("Class distribution")
plt.tight_layout()
plt.show()

#%% ---------------------------------- Split data ----------------------------------------------------------------------
from sklearn.model_selection import train_test_split

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# divide the data into training, validation and testing sets (70:15:15)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=SEED, stratify= y)
x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state=SEED, stratify= y_test)

#%% ---------------------------------- Save data -----------------------------------------------------------------------
dir = os.path.dirname('train/')
if not os.path.exists(dir):
    os.makedirs(dir)

np.save("train/x_train.npy", x_train); np.save("train/y_train.npy", y_train)
np.save("train/x_valid.npy", x_valid); np.save("train/y_valid.npy", y_valid)
np.save("train/x_test.npy", x_test); np.save("train/y_test.npy", y_test)