from functools import partial, reduce
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Lambda, Conv3D
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Input, concatenate, Add, Flatten, Reshape, Dropout
from tensorflow.keras.layers import GlobalAveragePooling3D, GlobalMaxPooling3D, MaxPooling3D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model

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
    
    x = __default_conv3D(input, filters=filters, strides=strides, weight_decay=5e-4)
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
        x = __default_conv3D(x, filters = filters, strides = strides, weight_decay=weight_decay)
        group_channel.append(x)

    group_merge = concatenate(group_channel, axis = -1)
    x = BatchNormalization()(group_merge)
    x = Activation('relu')(x)

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

def __bottleneck_layer(input, filters = 64, kernel_size = 3, strides = (1,1,1), cardinality = 16, weight_decay = 5e-4):
    '''
    Description: bottleneck layer for a single path(cardinality = 1)
    Args:   input: input tensor
            filters : number of filters for the last layer in a single path, suppose to be total number
                        of filters // cardinality of ResNeXt block.
            strides : strides, must be tuple of 3 elements
    '''
    x = input

    x = __default_conv3D(x, filters = filters // 2 // cardinality, kernel_size = 1, strides = strides)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = __default_conv3D(x, filters = filters // 2 // cardinality, kernel_size = kernel_size, strides = (1,1,1))
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = __default_conv3D(x, filters = filters, kernel_size = 1, strides = (1,1,1))
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

    x = __init_conv(input, filters=filters, strides=(2,2,2), weight_decay=weight_decay)

    for dep, filters in zip(depth, filter_list):
        for i in range(dep):
            strides = (2,2,2) if i == 0 else (1,1,1)
            x = __ResNeXt_block(x, filters = filters, strides=strides, cardinality = cardinality, weight_decay = weight_decay)

    x = GlobalAveragePooling3D()(x)
    x = Flatten()(x)
    x = Dense(5)(x)
    return x

def create_model_v2(input, filters = 8, weight_decay = 5e-4, dropout = 0.2):
    x = __init_split_conv(input, filters = filters)
    x = __default_conv3D(x, filters = 1024, strides=(2,2,2), weight_decay = weight_decay)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling3D()(x)
    x = Flatten()(x)
    y = []
     
    for i in range(5):
        _y = Dense(128)(x)
        _y = Dropout(dropout)(_y)
        _y = Dense(1)(_y)
        y.append(_y)
     
    y = concatenate(y, axis = -1)
     
    return y



if __name__ == "__main__":
    import numpy as np
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    input = Input(shape = (53, 63, 52, 53), batch_size = 4, dtype = tf.float32)
    output = create_model(input, filters = 128)
    model = Model(input, output)

    optimizer = keras.optimizers.RMSprop(0.001)
    model.compile(loss="mse",
        optimizer=optimizer,
        metrics=["mse", "mae"],
        experimental_run_tf_function=False)
    x = tf.constant(np.zeros(shape = (8, 53, 63, 52, 53), dtype = np.float32))
    y = tf.constant(np.zeros(shape = (8,5), dtype = np.float32))
    z = __init_grouped_conv(x, strides = (2,2,2))
    #model.fit(x,y,epochs = 3)
    #model.summary()
    