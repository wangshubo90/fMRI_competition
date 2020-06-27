import tensorflow as tf 
import tensorflow.keras as keras 
import pandas as pd 
import numpy as np 
import pickle, os, gc, glob
from tqdm import tqdm
from tensorflow.keras.models import Model

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

def calc_score(y, y_p):
    score = np.zeros(5)

    true_sum = y.sum(axis = 0)
    error = np.absolute(y-y_p)
    error = error.sum(axis = 0)

    score = error / true_sum
    weight = np.array([0.3, 0.175, 0.175, 0.175, 0.175])
    score = np.sum(score * weight)

    return

DATA_PATH = "../fMRI_train_pk"
df = pd.read_csv("train_scores.csv")
df = df.dropna().reset_index()

file_ls = []
y_ls = np.zeros(shape = (df.shape[0], 5))

loading_np = pickle.load(open('loading.pk', 'rb'))
fnc_np = pickle.load(open('fnc.pk', 'rb'))

for idx, row in df.iterrows():
    file_ls.append(os.path.join(DATA_PATH, str(int(row["Id"]))+".pk"))
    ys = row[2:].values
    y_ls[idx] = ys

y_true = np.array(y_ls)
y_pred = np.zeros(shape = (len(file_ls), 5), dtype = np.float32)

modeldir = '../saved model'
model_list = glob.glob(modeldir+'/*.h5')

for m in model_list:
    model = keras.models.load_model(m)
    i = 0
    for file, load, fnc in tqdm(list(zip(file_ls, loading_np, fnc_np))):
        f = None
        with open(file, 'rb') as f:
            img = pickle.load(f)
        
        img = normalize(img)
        y = model((img[np.newaxis], load.reshape((1,26)), fnc.reshape((1,1383))))
        y_pred[i] = np.squeeze(y)
        i += 1
        gc.collect()
        
        print(calc_score(y_true, y_pred))

