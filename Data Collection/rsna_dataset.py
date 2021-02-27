import numpy as np
import pandas as pd
import os
import pydicom
import cv2


#%% -------------------------------------- read metadata ---------------------------------------------------------------------------
df = pd.read_csv("rsna-dataset/stage_2_detailed_class_info.csv")

labels = ['Normal', 'Lung Opacity'] ## keep only normal and pneumonia cases
idx = df["class"].isin(labels)
df = df[idx]

#%% --------------------------------------- create directories --------------------------------------------------------------------
imgdir = 'rsna-dataset/stage_2_train_images/'  ## image folder
outdir = os.path.dirname('rsna-png/')  ## destination for PNG images
if not os.path.exists(outdir):
    os.makedirs(outdir)

rsna_dataset = os.path.dirname('../rsna_dataset/')  ## destination for classified images
if not os.path.exists(rsna_dataset):
    os.makedirs(rsna_dataset)

normal = os.path.dirname('../rsna_dataset/normal/')  ## normal cases
if not os.path.exists(normal):
    os.makedirs(normal)

lung = os.path.dirname('../rsna_dataset/pneumonia/')  ## pneumonia cases
if not os.path.exists(lung):
    os.makedirs(lung)

#%% ------------------------------------------ convert into PNG images ------------------------------------------------------
image_list = [f for f in os.listdir(imgdir)]

for file in image_list:
    ds = pydicom.read_file(imgdir + file)
    img = ds.pixel_array
    cv2.imwrite(outdir + "/" + file.replace('.dcm','.png'), img)

#%% ---------------------------------------- classify CXR images into normal and pneumonia ------------------------------------
import shutil
for (index, row) in df.iterrows():
    if row["class"] == "Normal":  ## normal cases
        filename = row["patientId"] + ".png"
        filepath = os.path.sep.join([outdir, filename])
        shutil.copy2(filepath, normal)  ## copy to normal folder
    else:  ## pneumonia cases
        filename = row["patientId"] + ".png"
        filepath = os.path.sep.join([outdir, filename])
        shutil.copy2(filepath, lung)  ## copy to pneumonia folder

print("Done.")
