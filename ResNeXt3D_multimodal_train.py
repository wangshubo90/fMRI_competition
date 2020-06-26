import os
import pandas as pd 
from functools import partial, reduce
import numpy as np 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import gc
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
    input_1, input_2, input_3 = input
    N = len(depth)
    filter_list = []
    
    for i in range(N):
        filter_list.append(filters * (2**i))

    x = __init_grouped_conv(input_1, filters=filters, strides=(2,2,2), weight_decay=weight_decay)

    for dep, filters in zip(depth, filter_list):
        for i in range(dep):
            strides = (2,2,2) if i == 0 else (1,1,1)
            x = __ResNeXt_block(x, filters = filters, strides=strides, cardinality = cardinality, weight_decay = weight_decay)
    
    x = GlobalAveragePooling3D()(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    #x = Dense(128, activation = 'relu', kernel_regularizer = keras.regularizers.l2(weight_decay))(x)
    #x = Dropout(0.3)(x)
    #x = Dense(64, activation = 'relu', kernel_regularizer = keras.regularizers.l2(weight_decay))(x)
    #x = Dropout(0.3)(x)
    #x = Dense(32, activation = 'relu', kernel_regularizer = keras.regularizers.l2(weight_decay))(x)
    #x = Dropout(0.3)(x)
    #x = Dense(5, activation = 'linear', kernel_regularizer = keras.regularizers.l2(weight_decay))(x)
    x = Dense(64, activation = 'relu', kernel_regularizer = keras.regularizers.l2(weight_decay))(x)

    y = Dense(64, activation = 'relu', kernel_regularizer = keras.regularizers.l2(weight_decay))(input_3)
    x = keras.layers.concatenate([x,y,input_2], axis = -1)
    output_1 = Dense(1, activation = 'linear', kernel_regularizer = keras.regularizers.l2(weight_decay), name = 'output_1')(x)
    output_2 = Dense(1, activation = 'linear', kernel_regularizer = keras.regularizers.l2(weight_decay), name = 'output_2')(x)
    output_3 = Dense(1, activation = 'linear', kernel_regularizer = keras.regularizers.l2(weight_decay), name = 'output_3')(x)
    output_4 = Dense(1, activation = 'linear', kernel_regularizer = keras.regularizers.l2(weight_decay), name = 'output_4')(x)
    output_5 = Dense(1, activation = 'linear', kernel_regularizer = keras.regularizers.l2(weight_decay), name = 'output_5')(x)

    return output_1, output_2, output_3, output_4, output_5

#================= Build Data pipeline =================
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
    img = img.transpose()
    return img

def FeatureGenerator(file_list,loading_np, fnc_np, augmentation = False):
    def generator():
        for file, load, fnc in zip(file_list, loading_np, fnc_np):
            gc.collect(2)
            #ith h5py.File(file, "r") as f:
                #img = f["SM_feature"][()]
            img = pickle.load(open(file, "rb"))
            img = normalize(img)

            #y = (i.reshape((1,)) for i in y)
            if augmentation:
                aug = random.randint(1,3)
                if aug == 3:
                    #img = tf.convert_to_tensor(img, dtype = tf.float32)
                    yield img, load, fnc
                else:
                    img = np.flip(img, axis = aug)
                    yield img, load, fnc
            else:
                yield img, load, fnc

    return generator

def LabelGenerator(y_list):
    def generator():
        for y in  y_list:
            y1 = y[0].reshape((1,))
            y2 = y[1].reshape((1,))
            y3 = y[2].reshape((1,))
            y4 = y[3].reshape((1,))
            y5 = y[4].reshape((1,))
            yield y1,y2,y3,y4,y5
    return generator
            
def DatasetReader(file_list, loading_np, fnc_np, y_list, shuffle_size, batch_size, augmentation = False):
    generator1 = FeatureGenerator(file_list, loading_np, fnc_np, augmentation=augmentation)
    dataset1 = tf.data.Dataset.from_generator(generator1,
                output_types = (tf.float32, )*3,
                output_shapes = (tf.TensorShape((53, 63, 52, 53)), 
                                    tf.TensorShape((26,)), 
                                    tf.TensorShape((1383,))
                                )
    )
    
    generator2 = LabelGenerator(y_list)
    dataset2 = tf.data.Dataset.from_generator(generator2,
                output_types = (tf.float32, )*5,
                output_shapes = (tf.TensorShape((1,)),)*5

    )

    dataset = tf.data.Dataset.zip((dataset1, dataset2))
    dataset = dataset.batch(batch_size).shuffle(shuffle_size).repeat().shard(hvd.size(), hvd.rank())
    
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

loading_np = pickle.load(open('loading.pk', 'rb'))
fnc_np = pickle.load(open('fnc.pk', 'rb'))

train_f, test_f, train_load, test_load, train_fnc, test_fnc, train_label, test_label = train_test_split(
    file_ls, loading_np, fnc_np, y_ls, test_size = 0.3, random_state = 42
)

val_f, evl_f, val_load, evl_load, val_fnc, evl_fnc, val_label, evl_label = train_test_split(
    test_f, test_load, test_fnc, test_label, test_size = 0.5, random_state = 42
)
'''
load_sc = StandardScaler()
fnc_sc = StandardScaler()

train_load = load_sc.fit_transform(train_load)
train_fnc = fnc_sc.fit_transform(train_fnc)

val_load = load_sc.transform(val_load)
val_fnc = fnc_sc.transform(val_fnc)

evl_load = load_sc.transform(evl_load)
evl_fnc = fnc_sc.transform(evl_fnc)
'''
BATCH_SIZE = 8
train_set = DatasetReader(train_f, train_load, train_fnc, train_label, 16, BATCH_SIZE, augmentation = True)
val_set = DatasetReader(val_f, val_load, val_fnc, val_label, 8, BATCH_SIZE // 2, augmentation = False)
evl_set = DatasetReader(evl_f, evl_load, evl_fnc, evl_label, 8, BATCH_SIZE // 2, augmentation = False)

#================== Configure Callbacks ==================
checkpoint_cb = keras.callbacks.ModelCheckpoint("./my_logs/multimodal/ResNeXt_ft128_dep22_w5-4_car16_{epoch}.h5", 
        monitor = 'val_loss', mode = 'min',
        save_best_only=True
        )

class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f} \n".format(logs["val_loss"] / logs["loss"]))

root_logdir = os.path.join(os.curdir, "./my_logs/multimodal")

def get_run_logdir(comment="_ResNeXt_ft128_dep22_w5-4_car16_NAdam0.0001_drop0.3_flip_continue"):
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
'''
input_1 = Input(shape = (53, 63, 52, 53), dtype = tf.float32, name = 'input_1')
input_2 = Input(shape = (26,), dtype = tf.float32, name = 'input_2')
input_3 = Input(shape = (1383), dtype = tf.float32, name = 'input_3')

input = (input_1, input_2, input_3)
output = create_model(input, filters = 128,depth=(2,2), cardinality=16, weight_decay = 5e-4)
model = Model(input, output)

optimizer = keras.optimizers.Nadam(
    learning_rate=0.0001 * hvd.size(), beta_1=0.9, beta_2=0.999, epsilon=1e-08
)

# set up Horovod
optimizer = hvd.DistributedOptimizer(optimizer)

model.compile(loss="mape",
        optimizer=optimizer,
        metrics=["mape"],
        loss_weights = [0.3, 0.175, 0.175, 0.175, 0.175],
        experimental_run_tf_function=False)
'''

model = keras.models.load_model('./my_logs/multimodal/ResNeXt_ft128_dep22_w5-4_car16.h5')

history = model.fit(train_set, steps_per_epoch= 256 // BATCH_SIZE, epochs=100,
          validation_data=val_set,
          validation_steps=256 // 4,
          callbacks=callbacks,
          verbose = 1 if hvd.rank() == 0 else 0
         )