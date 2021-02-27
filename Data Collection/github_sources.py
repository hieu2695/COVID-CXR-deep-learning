import torchxrayvision as xrv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

#%% --------------------- https://github.com/ieee8023/covid-chestxray-dataset ------------------------------------------
# the metadata and images are connected by 'filename' - the CXR image of the patient
print("Example from https://github.com/ieee8023/covid-chestxray-dataset:\n")
imgpath = 'covid-chestxray-dataset/images'  ## image folder
csvpath = 'covid-chestxray-dataset/metadata.csv' ## metadata folder

# read metadata as a dataframe
df = pd.read_csv(csvpath)

index = 10 ## select a random patient

# print the patient's information
print(df.loc[index,:])

# plot the CXR image
imgname = df.loc[index,:]["filename"] ## retrieve his/her CXR image
img = mpimg.imread(imgpath + "/" + imgname)
plt.imshow(img, cmap = 'gray')
plt.title("covid-chestxray-dataset example")
plt.show()


#%% ------------------------- https://github.com/agchung/Actualmed-COVID-chestxray-dataset -----------------------------
# the metadata and images are connected by 'imagename' - the CXR image of the patient
print("\nExample from https://github.com/agchung/Actualmed-COVID-chestxray-dataset:\n")
imgpath = 'Actualmed-COVID-chestxray-dataset/images'  ## image folder
csvpath = 'Actualmed-COVID-chestxray-dataset/metadata.csv' ## metadata folder

# read metadata as a dataframe
df = pd.read_csv(csvpath)

index = 10 ## select a random patient

# print the patient's information
print(df.loc[index,:])

# plot the CXR image
imgname = df.loc[index,:]["imagename"] ## retrieve his/her CXR image
img = mpimg.imread(imgpath + "/" + imgname)
plt.imshow(img, cmap = 'gray')
plt.title("Actualmed-COVID-chestxray-dataset example")
plt.show()

#%% ------------------------- https://github.com/agchung/Figure1-COVID-chestxray-dataset -----------------------------
# the metadata and images are connected by 'patientid'
print("\nExample from https://github.com/agchung/Figure1-COVID-chestxray-dataset:\n")
imgpath = 'Figure1-COVID-chestxray-dataset/images'  ## image folder
csvpath = 'Figure1-COVID-chestxray-dataset/metadata.csv' ## metadata folder

# read metadata as a dataframe
df = pd.read_csv(csvpath, encoding='ISO-8859-1')

index = 10 ## select a random patient

# print the patient's information
print(df.loc[index,:])

# plot the CXR image
imgname = df.loc[index,:]["patientid"] ## retrieve patientid
try:
    img = mpimg.imread(imgpath + "/" + imgname + ".jpg")
except:
    img = mpimg.imread(imgpath + "/" + imgname + ".png")

plt.imshow(img, cmap = 'gray')
plt.title("Figure1-COVID-chestxray-dataset example")
plt.show()
