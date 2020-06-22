import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd 

train_var_df = pd.read_csv('train_scores.csv')
loading_df = pd.read_csv('loading.csv')


with open('features_512.pk', 'rb') as f:
    x = pickle.load(f)

with open('y_train.pk', 'rb') as f:
    y = pickle.load(f)

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size = 0.3, random_state = 42
)

val_x, evl_x, val_y, evl_y = train_test_split(
    test_x, test_y, test_size = 0.5, random_state = 42
)

sc_X = StandardScaler()
sc_Y = StandardScaler()

X = sc_X.fit_transform(train_x)
Y = sc_Y.fit_transform(train_y)

model = MultiOutputRegressor(SVR(C = 0.075))
model.fit(X, Y)

def Kaggle_Score(y_true, y_pred):
    abs_err = abs(y_true-y_pred).sum(axis = 0)
    norm = y_true.sum(axis = 0)
    weight = np.array([0.3, 0.175, 0.175, 0.175, 0.175])
    score = np.sum(( abs_err / norm ) * weight)
    return score

def evaluate(val_x, val_y, model):
    val_Y = sc_Y.transform(val_y)
    val_X = sc_X.transform(val_x)
    val_Y_pred = model.predict(val_X)
    return Kaggle_Score(val_y, sc_Y.inverse_transform(val_Y_pred))

print('SVR results:')
print('Training score is : {:.4f}'.format(evaluate(train_x, train_y, model)))
print('Evaluation score is : {:.4f}'.format(evaluate(val_x, val_y, model)))
print('Validation score is : {:.4f}'.format(evaluate(evl_x, evl_y, model)))


DTRmodel = MultiOutputRegressor(DecisionTreeRegressor(max_depth=6))
DTRmodel.fit(X,Y)

print('DecisionTreeRegression results:')
print('Training score is : {:.4f}'.format(evaluate(train_x, train_y, DTRmodel)))
print('Evaluation score is : {:.4f}'.format(evaluate(val_x, val_y, DTRmodel)))
print('Validation score is : {:.4f}'.format(evaluate(evl_x, evl_y, DTRmodel)))
