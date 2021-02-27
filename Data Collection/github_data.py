import numpy as np
import pandas as pd
import os
import random
import shutil

#%% ----------------------------------------- Create directories ----------------------------------------------------------
datadir = os.path.dirname('../github_dataset/')
if not os.path.exists(datadir):
    os.makedirs(datadir)

outdir_covid = os.path.dirname('../github_dataset/covid/') ## covid cases
if not os.path.exists(outdir_covid):
    os.makedirs(outdir_covid)

outdir_pneumonia = os.path.dirname('../github_dataset/pneumonia/') ## pneumonia cases
if not os.path.exists(outdir_pneumonia):
    os.makedirs(outdir_pneumonia)

outdir_normal = os.path.dirname('../github_dataset/normal/') ## normal cases
if not os.path.exists(outdir_normal):
    os.makedirs(outdir_normal)

#%% ---------------------------------- image and metadata paths --------------------------------------------------------
path1 =  'covid-chestxray-dataset/'
image1 = path1 + 'images'
metadata1 = path1 + 'metadata.csv'

path2 =  'Figure1-COVID-chestxray-dataset/'
image2 = path2 + 'images'
metadata2 = path2 + 'metadata.csv'

path3 =  'Actualmed-COVID-chestxray-dataset/'
image3 = path3 + 'images'
metadata3 = path3 + 'metadata.csv'

#%% ---------------------------------- read metadata ----------------------------------------------------------------------
csv1 = pd.read_csv(metadata1)
csv2 = pd.read_csv(metadata2, encoding='ISO-8859-1')
csv3 = pd.read_csv(metadata3)

#%% ----------------------- create COVID dataset from cloned github repos----------------------------------------------------------------------
# covid-chestxray-dataset
views = ["PA", "AP", "AP Supine"]  ## keep only PA, AP and AP Supine view of CXR images
idx = csv1.view.isin(views)
csv1 = csv1[idx]

for (index, row) in csv1.iterrows():  ## loop over all rows
    if row["finding"] == "Pneumonia/Viral/COVID-19":  ## covid cases
        filename = row["filename"]  ## retrive the image name
        filepath = os.path.sep.join([image1, filename]) ## retrieve the image path
        shutil.copy2(filepath, outdir_covid)  ## copy the image to the covid folder
    elif row["finding"] == "No Finding": ## normal cases
        filename = row["filename"]
        filepath = os.path.sep.join([image1, filename])
        shutil.copy2(filepath, outdir_normal)
    elif row["finding"] == "todo": ## not normal, not pneumonia, not covid
        continue
    else:   ## other cases are pneumonia cases but non-covid
        ilename = row["filename"]
        filepath = os.path.sep.join([image1, filename])
        shutil.copy2(filepath, outdir_pneumonia)

# =====================================
# Figure1-COVID-chestxray-dataset
for (index, row) in csv2.iterrows():
    if row["finding"] == "COVID-19":
        filename = row["patientid"]  ## patientid + ".jpg" or ".png" will be the image name
        if os.path.exists(os.path.sep.join([image2, filename + '.jpg'])):
            filepath = os.path.sep.join([image2, filename + '.jpg'])
        elif os.path.exists(os.path.sep.join([image2, filename + '.png'])):
            filepath = os.path.sep.join([image2, filename + '.png'])
        shutil.copy2(filepath, outdir_covid)

    elif row["finding"] == "No finding":
        filename = row["patientid"]
        if os.path.exists(os.path.sep.join([image2, filename + '.jpg'])):
            filepath = os.path.sep.join([image2, filename + '.jpg'])
        elif os.path.exists(os.path.sep.join([image2, filename + '.png'])):
            filepath = os.path.sep.join([image2, filename + '.png'])
        shutil.copy2(filepath, outdir_normal)

    elif row["finding"] == "Pneumonia":
        filename = row["patientid"]
        if os.path.exists(os.path.sep.join([image2, filename + '.jpg'])):
            filepath = os.path.sep.join([image2, filename + '.jpg'])
        elif os.path.exists(os.path.sep.join([image2, filename + '.png'])):
            filepath = os.path.sep.join([image2, filename + '.png'])
        shutil.copy2(filepath, outdir_pneumonia)

    else:  ## there are nan values indicating unclear results
        continue

# =====================================
# Actualmed-COVID-chestxray-dataset
for (index, row) in csv3.iterrows():
    if row["finding"] == "COVID-19":
        filename = row["imagename"]
        filepath = os.path.sep.join([image3, filename])
        shutil.copy2(filepath, outdir_covid)
    elif row["finding"] == "No finding":
        filename = row["imagename"]
        filepath = os.path.sep.join([image3, filename])
        shutil.copy2(filepath, outdir_normal)
    else:
        continue








