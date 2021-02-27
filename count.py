import os

DIR = 'data/covid/'
count_covid =  len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

DIR = 'data/normal/'
count_normal =  len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

DIR = 'data/pneumonia/'
count_pneumonia =  len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

print("The number of covid samples is: ", count_covid)
print("The number of non-covid pneumonia samples is: ", count_pneumonia)
print("The number of normal samples is: ", count_normal)
