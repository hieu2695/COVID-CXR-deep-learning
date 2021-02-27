from PIL import Image
import os

#%% ------------------------------------- Create data directory to store CXR images ---------------------------------------
dir = os.path.dirname('data/')
if not os.path.exists(dir):
    os.makedirs(dir)

covid = os.path.dirname('data/covid/')  ## covid cases
if not os.path.exists(covid):
    os.makedirs(covid)

normal = os.path.dirname('data/normal/') ## normal cases
if not os.path.exists(normal):
    os.makedirs(normal)

pneumonia = os.path.dirname('data/pneumonia/') ## pneumonia cases
if not os.path.exists(pneumonia):
    os.makedirs(pneumonia)

#%% ----------------------------- Convert images into PNG and save them into 3 folders -------------------------------------

# CXR_dataset
## normal
imglist = [f for f in os.listdir('CXR_dataset/normal/')]
for f in imglist:
    sep = "."
    name = sep.join(f.split(sep)[:-1]) + ".png"
    img = Image.open('CXR_dataset/normal/' + f)
    img.save(normal + "/" + name)

## pneumonia
imglist = [f for f in os.listdir('CXR_dataset/pneumonia/')]
for f in imglist:
    sep = "."
    name = sep.join(f.split(sep)[:-1]) + ".png"
    img = Image.open('CXR_dataset/pneumonia/' + f)
    img.save(pneumonia + "/" + name)

# ==============================
# github_dataset
## covid
imglist = [f for f in os.listdir('github_dataset/covid/')]
for f in imglist:
    sep = "."
    name = sep.join(f.split(sep)[:-1]) + ".png"
    img = Image.open('github_dataset/covid/' + f)
    img.save(covid + "/" + name)

## normal
imglist = [f for f in os.listdir('github_dataset/normal/')]
for f in imglist:
    sep = "."
    name = sep.join(f.split(sep)[:-1]) + ".png"
    img = Image.open('github_dataset/normal/' + f)
    img.save(normal + "/" + name)

## pneumonia
imglist = [f for f in os.listdir('github_dataset/pneumonia/')]
for f in imglist:
    sep = "."
    name = sep.join(f.split(sep)[:-1]) + ".png"
    img = Image.open('github_dataset/pneumonia/' + f)
    img.save(pneumonia + "/" + name)

# ==============================
# COVID-19 Radiography
## covid
imglist = [f for f in os.listdir('COVID-19_radiography/covid/')]
for f in imglist:
    sep = "."
    name = sep.join(f.split(sep)[:-1]) + ".png"
    img = Image.open('COVID-19_radiography/covid/' + f)
    img.save(covid + "/" + name)

## normal
imglist = [f for f in os.listdir('COVID-19_radiography/normal/')]
for f in imglist:
    sep = "."
    name = sep.join(f.split(sep)[:-1]) + ".png"
    img = Image.open('COVID-19_radiography/normal/' + f)
    img.save(normal + "/" + name)

## pneumonia
imglist = [f for f in os.listdir('COVID-19_radiography/pneumonia/')]
for f in imglist:
    sep = "."
    name = sep.join(f.split(sep)[:-1]) + ".png"
    img = Image.open('COVID-19_radiography/pneumonia/' + f)
    img.save(pneumonia + "/" + name)

# ==============================
# RSNA dataset
## normal
imglist = [f for f in os.listdir('rsna_dataset/normal/')]
for f in imglist:
    sep = "."
    name = sep.join(f.split(sep)[:-1]) + ".png"
    img = Image.open('rsna_dataset/normal/' + f)
    img.save(normal + "/" + name)

## pneumonia
imglist = [f for f in os.listdir('rsna_dataset/pneumonia/')]
for f in imglist:
    sep = "."
    name = sep.join(f.split(sep)[:-1]) + ".png"
    img = Image.open('rsna_dataset/pneumonia/' + f)
    img.save(pneumonia + "/" + name)

print("Data generation process is completed.")