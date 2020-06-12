import os
import pandas as pd 
from functools import partial, reduce
import numpy as np 
import pickle
from sklearn.model_selection import train_test_split
#=================== Environment variables ===================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Lambda, Conv3D
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Input, concatenate, Add, Flatten, Dropout
from tensorflow.keras.layers import GlobalAveragePooling3D, GlobalMaxPooling3D, MaxPooling3D
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

def __init_conv(input, filters=64, strides=(1,1,1), weight_decay=5e-4):
    '''
    Description: initial convolutional layers before ResNeXt block
    Args:   input: input tensor
            filters: number of filters
            strides: strides, must be a tuple
            weight_decay: parameter for l2 regularization
    Return: output tensor
    '''
    
    x = __default_conv3D(input, filters=filters, strides=strides, weight_decay=weight_decay)
    x = BatchNormalization(axis = -1)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size = (2,2,2))(x)

    return x

def __init_grouped_conv(input, filters = 128, strides = (1,1,1), weight_decay = 5e-4):
    
    init = __default_conv3D(input, filters = filters - input.shape[-1] * 2, strides=strides, weight_decay=weight_decay)
    group_channel = [init]
    for i in range(input.shape[-1]):
        x = Lambda(lambda z:z[:, :, :, :, i])(input)
        x = tf.keras.backend.expand_dims(x, -1)
        x = __default_conv3D(x, filters = 2, strides = strides, weight_decay=weight_decay)
        group_channel.append(x)
    group_merge = concatenate(group_channel, axis = -1)
    x = BatchNormalization()(group_merge)
    x = Activation('relu')(x)

    return x
    
def __bottleneck_layer(input, filters = 64, kernel_size = 3, strides = (1,1,1), cardinality = 16, weight_decay = 5e-4):
    '''
    Description: bottleneck layer for a single path(cardinality = 1)
    Args:   input: input tensor
            filters : number of filters for the last layer in a single path, suppose to be total number
                        of filters // cardinality of ResNeXt block.
            strides : strides, must be tuple of 3 elements
    '''
    x = input

    x = __default_conv3D(x, filters = filters // 2 // cardinality, kernel_size = 1, strides = strides, weight_decay=weight_decay)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = __default_conv3D(x, filters = filters // 2 // cardinality, kernel_size = kernel_size, strides = (1,1,1), weight_decay=weight_decay)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = __default_conv3D(x, filters = filters, kernel_size = 1, strides = (1,1,1), weight_decay=weight_decay)
    x = BatchNormalization()(x)
    
    return x

def __ResNeXt_block(input, filters = 64, kernel_size = 3, strides = (1,1,1), cardinality = 16, weight_decay = 5e-4):
    '''
    Description: refer to the ResNeXt architechture. One ResNeXt_block contains several paths (cardinality) of bottleneck layers joint by a skip connection.
    '''

    if strides[0] == 1:
        init = input
    elif strides[0] > 1:
        init = __default_conv3D(input, filters = filters, kernel_size=kernel_size, strides=strides, weight_decay = weight_decay)
        init = BatchNormalization()(init)

    x = [init]
    
    for i in range(cardinality):
        x_sub = __bottleneck_layer(input, filters = filters, kernel_size=kernel_size, strides=strides, cardinality=cardinality, weight_decay=weight_decay)
        x_sub = BatchNormalization()(x_sub)
        x.append(x_sub)

    x = Add()(x)
    x = Activation('relu')(x)

    return x

def create_model(input, filters = 64, depth = (2,2,2), cardinality = 16, weight_decay = 5e-4):
    '''
    Description:
    Args:   input: input tf tensor
            filters: filter numbers of initial convolutional layer and first chunk ResNeXt blocks. Filter number doubles there after
            depth: a tuple of number of ResNeXt blocks for each step of feature map resolution.
            cardinality: number of bottleneck layer paths
            weight_decay: l2 regularization parameter
    Return: output: output tf tensor
    '''
    N = len(depth)
    filter_list = []
    
    for i in range(N):
        filter_list.append(filters * (2**i))

    x = __init_grouped_conv(input, filters=filters, strides=(2,2,2), weight_decay=weight_decay)

    for dep, filters in zip(depth, filter_list):
        for i in range(dep):
            strides = (2,2,2) if i == 0 else (1,1,1)
            x = __ResNeXt_block(x, filters = filters, strides=strides, cardinality = cardinality, weight_decay = weight_decay)
    
    x = GlobalAveragePooling3D()(x)
    x = Flatten()(x)
    x = Dense(64, activation = 'relu', kernel_regularizer = keras.regularizers.l2(weight_decay))(x)
    x = Dropout(0.3)(x)
    x = Dense(5, activation = 'linear', kernel_regularizer = keras.regularizers.l2(weight_decay))(x)
    return x

def __init_split_conv(input, filters = 8, strides = (1,1,1), weight_decay = 5e-4):
    
    group_channel = []
    for i in range(input.shape[-1]):
        x = Lambda(lambda z:z[:, :, :, :, i])(input)
        x = tf.keras.backend.expand_dims(x, -1)
        x = __default_conv3D(x, filters = filters, strides = strides, weight_decay=weight_decay)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x_orig = __default_conv3D(x, kernel_size=1, filters = filters, strides = (2,2,2), weight_decay=weight_decay)

        x = __default_conv3D(x, kernel_size=1, filters = filters // 2, strides = (2,2,2), weight_decay=weight_decay)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = __default_conv3D(x, kernel_size=3, filters = filters, strides = (1,1,1), weight_decay=weight_decay)     

        x = x + x_orig
        x = BatchNormalization()(x)
        x = Activation('relu')(x)  

        group_channel.append(x)

    group_merge = concatenate(group_channel, axis = -1)
    x = BatchNormalization()(group_merge)
    x = Activation('relu')(x)

    return x

def create_model_v2(input, filters = 8, strides = (2,2,2), weight_decay = 5e-4, dropout = 0.2, last_layer = None):
    x = __init_split_conv(input, filters = filters, strides = strides)
    x = __default_conv3D(x, filters = 1024, strides=(2,2,2), weight_decay = weight_decay)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling3D()(x)
    x = Flatten()(x)
    if last_layer == 'one':
        y = Dense(256)(x)
        y = Dropout(dropout)(y)
        y = Dense(5)(y)
    elif last_layer == 'split':
        y = []
     
        for i in range(5):
            _y = Dense(128)(x)
            _y = Dropout(dropout)(_y)
            _y = Dense(1)(_y)
            y.append(_y)
        
        y = concatenate(y, axis = -1)
    
    return y
#================= Build Data pipeline =================
def normalize(img):
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
checkpoint_cb = keras.callbacks.ModelCheckpoint("./my_logs/ResNeXt_3gpu_groupinit_l2_5e-3_dep222_dropout.h5", 
        monitor = 'val_mse', mode = 'min',
        save_best_only=True
        )

class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f} \n".format(logs["val_loss"] / logs["loss"]))

root_logdir = os.path.join(os.curdir, "./my_logs/ResNeXt_3gpu_groupinit_l2_5e-3_dep222_dropout")

def get_run_logdir(comment=""):
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S{}".format(comment))
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

callbacks = [
    tensorboard_cb, checkpoint_cb
]

#================== Training ==================
input = Input(shape = (53, 63, 52, 53), dtype = tf.float32)
output = create_model_v2(input, filters = 8, strides = (2,2,2), weight_decay=5e-4, dropout = 0.2, last_layer = 'split')
model = Model(input, output)

optimizer = keras.optimizers.RMSprop(0.001)

model.compile(loss="mse",
        optimizer=optimizer,
        metrics=["mse", "mae"],
        experimental_run_tf_function=False)

history = model.fit(train_set, steps_per_epoch= 256 // BATCH_SIZE, epochs=150,
          validation_data=val_set,
          validation_steps=800 // 4,
          verbose = 1
         )