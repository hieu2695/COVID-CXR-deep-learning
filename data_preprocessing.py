import os
import numpy as np
import cv2

#%% ------------------------------------------ Data directories -------------------------------------------------------------

COVID_DIR = "data/covid/"
NOR_DIR = "data/normal/"
PNEU_DIR = "data/pneumonia/"

#%% ------------------------------------------------ Load data ---------------------------------------------------------
x, y = [], [] ## store images and their labels
RESIZE_TO = 128 ## resize images to 128x128 px

# covid CXR
for path in os.listdir(COVID_DIR) :
    try:
        image = cv2.resize(cv2.imread(COVID_DIR + path),  (RESIZE_TO, RESIZE_TO))
        x.append(image)
        y.append("COVID")
    except:
        pass

# normal CXR
for path in os.listdir(NOR_DIR) :
    try:
        image = cv2.resize(cv2.imread(NOR_DIR + path),  (RESIZE_TO, RESIZE_TO))
        x.append(image)
        y.append("NORMAL")
    except:
        pass


# pneumonia CXR
for path in os.listdir(PNEU_DIR) :
    try:
        image = cv2.resize(cv2.imread(PNEU_DIR + path).astype(np.float32),  (RESIZE_TO, RESIZE_TO))
        x.append(image)
        y.append("PNEUMONIA")
    except:
        pass


#%% ------------------------------------------------- Remove duplicates ----------------------------------------------------------
unique_x, unique_y = [], []

for i in range(len(x)):
    if not any(np.array_equal(x[i], arr) for arr in unique_x):
        unique_x.append(x[i])
        unique_y.append(y[i])


#%% --------------------------
dir = os.path.dirname('COVID_npy/')
if not os.path.exists(dir):
    os.makedirs(dir)


unique_x, unique_y = np.array(unique_x), np.array(unique_y)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(["COVID", "NORMAL", "PNEUMONIA"])
unique_y = le.transform(unique_y)

np.save("COVID_npy/input.npy", unique_x)
np.save("COVID_npy/target.npy", unique_y)

print("Completed.")


