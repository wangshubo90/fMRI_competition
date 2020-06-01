#!/home/blue/tf2/bin/python
import os
import pandas as pd
from functools import partial
import numpy as np
import pickle

#================ Environment variables ================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras

#================ Load custom layers ===================
DefaultConv3D = partial(keras.layers.Conv3D, kernel_size=3, strides=(1,)*3,
        padding="SAME", use_bias=True, kernel_regularizer = keras.regularizers.l2(0.01))

class ResidualUnit(keras.layers.Layer):
    # separate construction and execution
    # be aware of the strides' shape
    def __init__(self, filters, strides=(1,)*3, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.filters = filters
        self.strides = strides
                
        # a list a layers that can be iterated
        self.main_layers = [
                DefaultConv3D(self.filters, strides=self.strides, kernel_initializer="he_normal"),
                keras.layers.BatchNormalization(),
                self.activation,
                DefaultConv3D(self.filters, strides=(1,)*3, kernel_initializer="he_normal"),
                keras.layers.BatchNormalization()
                ]
        self.skip_layers = []
        if np.prod(self.strides) > 1:
            #self.skip_layers = [keras.layers.MaxPool3D(pool_size=(2,)*3, strides=strides, padding="SAME")]
            
            self.skip_layers = [
                DefaultConv3D(self.filters, kernel_size=1, strides=self.strides, kernel_initializer="he_normal"),
                keras.layers.BatchNormalization()
                ]          
            
    def call(self, inputs, **kwargs):
        x = inputs
        orig_x = inputs
        
        for layer in self.main_layers:
            x = layer(x) # f(x)
        
        for layer in self.skip_layers:
            orig_x = layer(orig_x)
        
        return self.activation(x + orig_x)
    
    def get_config(self):
        config = super(ResidualUnit, self).get_config()
        config.update({'filters': self.filters, 'strides':self.strides})
        
        return config

#=========================== Load model ===========================
model = keras.models.load_model("my_logs/First_try.h5", custom_objects = {"ResidualUnit": ResidualUnit})

#=========================== Read data ============================
DATA_PATH = "../fMRI_train_pk"
df = pd.read_csv("train_scores.csv")
df = df.dropna()

file_ls = []
y_ls = []

for _, row in df.iterrows():
    file_ls.append(os.path.join(DATA_PATH, str(int(row["Id"]))+".pk"))
    ys = [item for _, item in row.iteritems()]
    y_ls.append(ys[1:])

y_ls = np.array(y_ls, dtype = np.float32)

y_pre = np.zeros(shape = y_ls.shape, dtype = np.float32)

i = 0

def normalize(img):
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    img = img.transpose()
    return img

while file_ls:
    j = 0
    x_mini = []
    while j < 16 and file_ls:
        file = file_ls.pop(0)
        with open(file, 'rb') as f:
            x = pickle.load(f)
            x = normalize(x)
        x_mini.append(x)
        j += 1

    x_mini = np.array(x_mini)
    y = model.predict(x_mini)
    print(y.shape)
    y_pre[i:i+j] = y
    i += 16

    