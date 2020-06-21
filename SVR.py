from sklearn.model_selection import train_test_split
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

with open('features_512.pk', 'rb') as f:
    x = pickle.load(f)

with open('y_train.pk', 'rb') as f:
    y = pickle.load(f)

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size = 0.3, random_state = 42
)

val_x, val_x, evl_y, evl_y = train_test_split(
    x, y, test_size = 0.3, random_state = 42
)

sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(train_x)
y = sc_y.fit_transform(train_y)