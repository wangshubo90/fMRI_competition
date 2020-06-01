#!/home/blue/tf2/bin/python
import os
import pandas as pd
from functools import partial
import numpy as np
import SimpleITK as sitk # to read nii files
from sklearn.model_selection import train_test_split
import pickle
#================ Environment variables ================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
import horovod.tensorflow.keras as hvd
#=================== Set up Horovod =================
# comment out this chunk of code if you train with 1 gpu
hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

#================= Build Data pipeline =================
def normalize(img):
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    img = img.transpose()
    return img

def DataGenerator(file_list, y_list):
    def generator():
        for file, y in zip(file_list, y_list):
            #ith h5py.File(file, "r") as f:
                #img = f["SM_feature"][()]
            img = pickle.load(open(file, "rb"))
            img = normalize(img)
            yield img, y

    return generator
            
def DatasetReader(file_list, y_list, shuffle_size, batch_size):
    generator = DataGenerator(file_list, y_list)
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types = (tf.float32, tf.float32),
        output_shapes = (tf.TensorShape((53, 63, 52, 53)), tf.TensorShape((5,)))
    )
    
    dataset = dataset.batch(batch_size).shuffle(shuffle_size).repeat()
    
    #return dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset.prefetch(1)

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



train_f, test_f, train_label, test_label = train_test_split(
    file_ls, y_ls, test_size = 0.3, random_state = 42
)

val_f, evl_f, val_label, evl_label = train_test_split(
    test_f, test_label, test_size = 0.5, random_state = 42
)

BATCH_SIZE = 8
train_set = DatasetReader(train_f, train_label, 16, BATCH_SIZE)
val_set = DatasetReader(val_f, val_label, 8, BATCH_SIZE // 2)
evl_set = DatasetReader(evl_f, evl_label, 8, BATCH_SIZE // 2)

#==================== Build model ====================
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

filters = (16, 32, 64)
strides = (1, 2, 2)
#(1,1,1)
model = keras.models.Sequential()
model.add(DefaultConv3D(filters[0], kernel_size=3, strides=(1,)*3,
        input_shape=[53, 63, 52, 53], kernel_initializer="he_normal"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool3D(pool_size=(2,)*3, padding="SAME"))

for filter, stride in zip(filters[1:], strides[1:]):
    model.add(ResidualUnit(filter, strides=(stride,)*3))
    model.add(ResidualUnit(filter, strides=(1,)*3))

model.add(keras.layers.GlobalAvgPool3D())
model.add(keras.layers.Flatten()) # 128 
model.add(keras.layers.Dense(16, activation="relu", kernel_regularizer = keras.regularizers.l2(0.002)))
#model.add(keras.layers.Dropout(0.5 ))
model.add(keras.layers.Dense(5))
optimizer = keras.optimizers.RMSprop(0.001)

# set up Horovod
optimizer = hvd.DistributedOptimizer(optimizer)

model.compile(loss="mse",
        optimizer=optimizer,
        metrics=["mse", "mae"],
        experimental_run_tf_function=False)

#================== Configure Callbacks ==================
checkpoint_cb = keras.callbacks.ModelCheckpoint("./my_logs/First_try.h5", 
        monitor = 'val_loss', mode = 'min',
        save_best_only=True
        )

class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f} \n".format(logs["val_loss"] / logs["loss"]))

root_logdir = os.path.join(os.curdir, "./my_logs/First_try")

def get_run_logdir(comment=""):
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S{}".format(comment))
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0)
]

if hvd.rank() == 0:
    callbacks.append(tensorboard_cb)
    callbacks.append(checkpoint_cb)
#================== Training ==================
history = model.fit(train_set, steps_per_epoch= 128 // BATCH_SIZE, epochs=500,
          validation_data=val_set,
          validation_steps=800 // 4,
          callbacks=callbacks,
          verbose = 1 if hvd.rank() == 0 else 0
         )