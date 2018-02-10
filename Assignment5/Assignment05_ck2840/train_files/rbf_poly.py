import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.sparse import csc_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
import pickle

def euclidean_dist_matrix(data_1, data_2):
    """
    Returns matrix of pairwise, squared Euclidean distances
    """
    norms_1 = (data_1 ** 2).sum(axis=1)
    norms_2 = (data_2 ** 2).sum(axis=1)
    data_2 = np.transpose(data_2)
    prod = np.dot(data_1, data_2)
    prod = 2*prod
    norms_1 = norms_1.reshape(-1,1)
    sum = norms_1 + norms_2
    sum = sum -  prod
    abs_mat = np.abs(sum)
    return abs_mat

def prbf_kernel(data_1, data_2):
    #Combination of Polynomial + RBF Kernel
    gamma = 0.05
    dists_sq = euclidean_dist_matrix(data_1, data_2)
    z = (10+np.exp(-gamma * dists_sq))
    return z

# import some data to play with
X_D= pd.read_csv('dataset1.csv').as_matrix()

Y = X_D[:,0]
X = np.delete(X_D, 0, 1 )

#Split test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.3, random_state=42 )

clf = SVC(kernel=prbf_kernel, C=5)
clf.fit(X_train, Y_train)
Z = clf.predict(X_test)
print(accuracy_score(Y_test,Z))

# writing out the pickled model

fp = open("clf.pkl", 'wb')
pickle.dump(clf,fp)
fp.close()   
