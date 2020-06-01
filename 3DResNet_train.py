import tensorflow as tf
import os
import pandas as pd
from tensorflow import keras
from functools import partial
import numpy as np
import SimpleITK as sitk # to read nii files
from sklearn.model_selection import train_test_split
#================ Environment variables ================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
#================= Build Data pipeline =================
def RotateAugmentation(img):
    '''
    Description: rotate the input 3D image along z-axis by random angles 
    Args:
        img: sitk.Image, dimension is 3
    Return: sitk.Image
    '''
    img.SetOrigin((0,0,0))
    img.SetSpacing((1.0, 1.0, 1.0))
    dim = img.GetSize()
    translation = [np.random.randint(0, 5), np.random.randint(0, 5), 0]
    angle = np.float64(np.random.randint(0, 360) / 180 * np.pi)
    rotation = [np.float64(0), np.float64(0), angle] # x-axis, y-axis, z-axis angles
    center = [(d-1) // 2 for d in dim]
    transformation = sitk.Euler3DTransform()    
    transformation.SetCenter(center)            # SetCenter takes three args
    transformation.SetRotation(*rotation)         # SetRotation takes three args
    transformation.SetTranslation(translation)
    aug_img = sitk.Resample(sitk.Cast(img, sitk.sitkFloat32), 
            sitk.Cast(img[0:96, 0:96, :], 
            sitk.sitkFloat32), transformation, 
            sitk.sitkLinear, 0.0, sitk.sitkFloat32
            )
    return aug_img

def IdentityAugmentation(img):
    return img[2:98, 2:98, :]

def DataGenerator(file_ls, label_ls, augmentation = None, example_n = 2, special_multiplier = 1):
    def normalize(img):
        if type(img) == sitk.Image:
            img = sitk.GetArrayFromImage(img)
        mean = np.mean(img)
        std = np.std(img)
        img = (img - mean) / std
        img = img[:,:,:,np.newaxis].astype(np.float32)
        return img

    def generator():
        for f, y in zip(file_ls, label_ls):
            y = np.array([y]).astype(np.float32)
            img = sitk.ReadImage(os.path.join(datapath, f))
            orig_img = img[2:98, 2:98, :]
            if augmentation is not None:
                if y == 0:
                    imgls = [normalize(orig_img)]
                    N = example_n
                    for _ in range(N):
                        imgls.append(normalize(augmentation(img)))

                    for img in imgls:
                        yield img, y

                elif y > 0 :
                    imgls = [normalize(orig_img)]
                    N = example_n * special_multiplier
                    for _ in range(N):
                        imgls.append(normalize(augmentation(img)))

                    for img in imgls:
                        yield img, y
        
            else:
                yield normalize(orig_img), y

    return generator

def DatasetReader(file_ls, label_ls, shuffle_size, batch_size, augmentation = None, example_n = 2, special_multiplier = 1):
    generator = DataGenerator(file_ls, label_ls, augmentation=augmentation, 
            example_n = example_n, special_multiplier = special_multiplier
    )
    dataset = tf.data.Dataset.from_generator(generator,
        output_types = (tf.float32, tf.float32),
        output_shapes = (tf.TensorShape((48, 96, 96, 1)), tf.TensorShape((1)))
    )
    dataset = dataset.repeat()
    dataset = dataset.shuffle(shuffle_size).batch(batch_size)

    return dataset.prefetch(tf.data.experimental.AUTOTUNE)

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
        input_shape=[48, 96, 96, 1], kernel_initializer="he_normal"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool3D(pool_size=(2,)*3, padding="SAME"))

for filter, stride in zip(filters[1:], strides[1:]):
    model.add(ResidualUnit(filter, strides=(stride,)*3))
    model.add(ResidualUnit(filter, strides=(1,)*3))

model.add(keras.layers.GlobalAvgPool3D())
model.add(keras.layers.Flatten()) # 128 
#model.add(keras.layers.Dense(32, activation="relu", kernel_regularizer = keras.regularizers.l2(0.002)))
#model.add(keras.layers.Dropout(0.5 ))
model.add(keras.layers.Dense(4, activation="softmax", kernel_regularizer = keras.regularizers.l2(2)))
#model.add(keras.layers.Dropout(0.2 ))
optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"])

#================== Configure Callbacks ==================
#_{epoch}
checkpoint_cb = keras.callbacks.ModelCheckpoint("./my_logs/Quad_rotate_aug_Classes.h5", 
        monitor = 'val_accuracy', mode = 'max',
        save_best_only=True
        )

class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f} \n".format(logs["val_loss"] / logs["loss"]))

root_logdir = os.path.join(os.curdir, "./my_logs/Quad_rotate_aug_Classes")

def get_run_logdir(comment=None):
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S{}".format(comment))
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

#==================== Training ====================
np.random.seed(42)
datapath = r'/home/spl/Hands-on-machine-learning/Codes/100x100x48 niis'
file_df = pd.read_csv('File_reference.csv')

file_ls = file_df['File name'].to_numpy()
label = file_df['Group'].to_numpy()
label = label.astype('int64')

train_img, test_img, train_l, test_l = train_test_split(
        file_ls, label, test_size=0.3, random_state=42
)
#del imgs
# split test data into validation and evaluation evenly
val_img, evl_img, val_l, evl_l = train_test_split(
        test_img, test_l, test_size = 0.5, random_state=42
)

print(train_img.shape)
print(val_img.shape)
print(evl_img.shape)

print(sum(train_l >= 1) / train_l.shape[0])
print(sum(val_l >= 1) / val_l.shape[0])
print(sum(evl_l >= 1) / evl_l.shape[0])

BATCH_SIZE = 36

train_set = DatasetReader(train_img, train_l, 640, BATCH_SIZE,
        augmentation = RotateAugmentation, example_n = 1, special_multiplier = 2
)
val_set = DatasetReader(val_img, val_l, 180, BATCH_SIZE) #, augmentation = IdentityAugmentation, example_n = 1, special_multiplier = 1
evl_set = DatasetReader(evl_img, evl_l, 180, BATCH_SIZE)

history = model.fit(train_set, steps_per_epoch= 640 // BATCH_SIZE, epochs=100,
          validation_data=val_set,
          validation_steps=180 // BATCH_SIZE,
          callbacks=[checkpoint_cb,  
                     PrintValTrainRatioCallback(), tensorboard_cb]
         )

model = keras.models.load_model("./my_logs/Quad_rotate_aug_Classes.h5", custom_objects={"ResidualUnit": ResidualUnit})
model.evaluate(evl_set, steps= 180 // BATCH_SIZE)