# import some data to play with
from sklearn.metrics import accuracy_score
from predict import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
X_D= pd.read_csv('dataset1.csv').as_matrix()
Y = X_D[:,0]
X = np.delete(X_D, 0, 1)
#split test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.3, random_state=42 )
Z = predictSVM(X_test)
print(accuracy_score(Y_test,Z))
