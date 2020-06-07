import numpy as np
import tensorflow.keras as keras
import pickle

class DataGenerator(keras.utils.Sequence):
    def __init__(self, file_list, y_list,shape = (53, 63, 52, 53), batch_size = 32):
        self.file_list = file_list
        self.y_list = y_list
        self.batch_size = batch_size
        self.dim = shape
        self.indexes = np.arange(len(self.file_list))

    def __len__(self):
        return int(np.floor(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        x, y = self.__data_generation(indexes)
        return x, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_list))

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros(shape = (self.batch_size, *self.dim), dtype = np.float32)
        y = np.zeros(shape = (self.batch_size, 5), dtype = np.float32)
        # Generate data
        for i in indexes:
            # Store sample
            x = pickle.load(open(self.file_list[i], 'rb'))
            for j in range(x.shape[0]):
                mean = np.mean(x[j,])
                std = np.std(x[j,])
                if std == 0.0:
                    pass
                else:
                    x[j,] = ( x[j,] - mean ) / std
            x = x.transpose()
            # Store class
            X[i,] = x

            y[i,] = self.y_list[i,]

        return X, y