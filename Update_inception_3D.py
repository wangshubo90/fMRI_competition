import os
import pandas as pd 
from functools import partial, reduce
import numpy as np 
import pickle
from sklearn.model_selection import train_test_split

import cv2 
import math 
from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler
#=================== Environment variables ===================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Lambda, Conv3D, Dropout
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Input, concatenate, Add, Flatten
from tensorflow.keras.layers import GlobalAveragePooling3D, GlobalMaxPooling3D, MaxPooling3D, AveragePooling3D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
import horovod.tensorflow.keras as hvd
#======================= Set up Horovod ======================
# comment out this chunk of code if you train with 1 gpu
hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
#========================= Build Model =======================    
def __default_conv3D(input, filters=8, kernel_size=3, strides=(1,1,1), weight_decay = 1e-4, **kwargs):
    '''
    Description: set up defaut parameters for Conv3D layers
    '''
    DefaultConv3D = partial(
        keras.layers.Conv3D, 
        filters = filters,
        kernel_size=kernel_size, 
        strides=strides,
        padding="SAME", 
        use_bias=True, 
        kernel_regularizer = keras.regularizers.l2(weight_decay),
        kernel_initializer="he_normal",
        **kwargs
    )
    return DefaultConv3D()(input)

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv3D(filters_1x1, 1, padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    
    conv_3x3 = Conv3D(filters_3x3_reduce, 1, padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv3D(filters_3x3, 3, padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv3D(filters_5x5_reduce, 1, padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv3D(filters_5x5, 5, padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPooling3D(3, strides=(1, 1,1), padding='same')(x)
    pool_proj = Conv3D(filters_pool_proj, 1, padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=4, name=name)
    
    return output

kernel_init = tf.keras.initializers.glorot_uniform()
bias_init = tf.keras.initializers.Constant(value=0.2)
input_layer = Input(shape=(53, 63, 52, 53))

x = Conv3D(64, 3, padding='same', strides=1, activation='relu', name='conv_1_7x7', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
x = MaxPooling3D(3, padding='same', strides=1, name='max_pool_1_3x3')(x)
#x = Conv3D(64, 1, padding='same', strides=(1, 1,1), activation='relu', name='conv_2a_3x3/1')(x)
#x = Conv3D(192, 3, padding='same', strides=(1, 1,1), activation='relu', name='conv_2b_3x3/1')(x)
#x = MaxPooling3D(3, padding='same', strides=(2, 2,2), name='max_pool_2_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=64,
                     filters_3x3_reduce=96,
                     filters_3x3=128,
                     filters_5x5_reduce=16,
                     filters_5x5=32,
                     filters_pool_proj=32,
                     name='inception_3a')

x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=192,
                     filters_5x5_reduce=32,
                     filters_5x5=96,
                     filters_pool_proj=64,
                     name='inception_3b')

x = MaxPooling3D(3, padding='same', strides=(2, 2,2), name='max_pool_3_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=192,
                     filters_3x3_reduce=96,
                     filters_3x3=208,
                     filters_5x5_reduce=16,
                     filters_5x5=48,
                     filters_pool_proj=64,
                     name='inception_4a')


x1 = AveragePooling3D(5, strides=3)(x)
x1 = Conv3D(128, 1, padding='same', activation='relu')(x1)
x1 = Flatten()(x1)
x1 = Dense(1024, activation='relu')(x1)
x1 = Dropout(0.7)(x1)
x1 = Dense(10, activation='softmax', name='auxilliary_output_1')(x1)

x = inception_module(x,
                     filters_1x1=160,
                     filters_3x3_reduce=112,
                     filters_3x3=224,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4b')

x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=256,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4c')

x = inception_module(x,
                     filters_1x1=112,
                     filters_3x3_reduce=144,
                     filters_3x3=288,
                     filters_5x5_reduce=32,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4d')


x2 = AveragePooling3D(5, strides=3)(x)
x2 = Conv3D(128, 1, padding='same', activation='relu')(x2)
x2 = Flatten()(x2)
x2 = Dense(1024, activation='relu')(x2)
x2 = Dropout(0.7)(x2)
x2 = Dense(10, activation='softmax', name='auxilliary_output_2')(x2)

x = inception_module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_4e')

x = MaxPooling3D(3, padding='same', strides=(2, 2,2), name='max_pool_4_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_5a')

x = inception_module(x,
                     filters_1x1=384,
                     filters_3x3_reduce=192,
                     filters_3x3=384,
                     filters_5x5_reduce=48,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_5b')

x = GlobalAveragePooling3D(name='avg_pool_5_3x3/1')(x)

x = Dropout(0.4)(x)

x = Dense(5, activation='softmax', name='output')(x)
model = Model(input_layer, [x, x1, x2], name='inception_v1')
model.summary()


input = Input(shape = (53, 63, 52, 53), dtype = tf.float32)
output = create_model(input, weight_decay=5e-3)
model = Model(input, output)

optimizer = keras.optimizers.RMSprop(0.001 * hvd.size())

# set up Horovod
optimizer = hvd.DistributedOptimizer(optimizer)

model.compile(loss=['mse', 'mse', 'mse'],
        optimizer=optimizer,
        metrics=["mse", "mae"],
        experimental_run_tf_function=False)

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

#================== Configure Callbacks ==================
checkpoint_cb = keras.callbacks.ModelCheckpoint("./my_logs/ResNeXt_3gpu_linear_act.h5", 
        monitor = 'val_loss', mode = 'min',
        save_best_only=True
        )

class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f} \n".format(logs["val_loss"] / logs["loss"]))

root_logdir = os.path.join(os.curdir, "./my_logs/ResNeXt_3gpu_linear_act")

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
history = model.fit(train_set[:,0],[train_set[:,1:4],train_set[:,1:4],train_set[:,1:4]], steps_per_epoch= 256 // BATCH_SIZE, epochs=100,
          validation_data=val_set[:,0],[val_set[:,1:4],val_set[:,1:4],val_set[:,1:4]]
          validation_steps=800 // 4,
          callbacks=callbacks,
          verbose = 1 if hvd.rank() == 0 else 0
         )

