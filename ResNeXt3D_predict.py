import tensorflow as tf 
import tensorflow.keras as keras 
import pandas as pd 
import numpy as np 
import pickle
from tqdm import tqdm
import os
from tensorflow.keras.models import Model
import gc

model = keras.models.load_model('./my_logs/ResNeXt_grouped128_dep22_w5-3_no_dropout_retry_NADAM_dep222_car32.h5')
new_model = Model(model.input, outputs = model.layers[-3].output)

def normalize_channel(img):
    shape = img.shape
    for i in range(shape[0]):
        map = img[i,:,:,:]
        mean = np.mean(map)
        std = np.std(map)
        if std == 0.0:
            pass
        else:
            img[i,:,:,:] = (map - mean) / std
    img = img.transpose()
    return img

def normalize(img):
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    return img.transpose()

DATA_PATH = "../fMRI_train_pk"
df = pd.read_csv("train_scores.csv")
df = df.dropna().reset_index()

file_ls = []
y_ls = np.zeros(shape = (df.shape[0], 5))

for idx, row in df.iterrows():
    file_ls.append(os.path.join(DATA_PATH, str(int(row["Id"]))+".pk"))
    ys = row[2:].values
    y_ls[idx] = ys

y_pred = np.zeros(shape = (len(file_ls), 512), dtype = np.float32)
i = 0
#for file in tqdm(file_ls):
for file in file_ls:
    f = None
    with open(file, 'rb') as f:
        img = pickle.load(f)
    
    img = normalize_channel(img)
    y = new_model.predict(img[np.newaxis])
    y_pred[i] = np.squeeze(y)
    i += 1
    gc.collect()
'''
with open('features_512.pk', 'wb') as f:
    pickle.dump(y_pred, f)

with open('y_train.pk', 'wb') as f:
    pickle.dump(y_pred, f)
'''

################
y  = np.array(y_ls)
y_p = np.squeeze(np.array(y_pred))

score = np.zeros(5)

true_sum = y.sum(axis = 0)
error = np.absolute(y-y_p)
error = error.sum(axis = 0)

score = error / true_sum
weight = np.array([0.3, 0.175, 0.175, 0.175, 0.175])
score = np.sum(score * weight)

print(score)