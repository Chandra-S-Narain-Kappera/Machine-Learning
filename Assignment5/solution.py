import numpy as np
import pickle
import csv

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

#read from file and convert it to an numpy array.
data = []
label = []
with open('dataset1.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')
	for row in reader:
		label.append(row[0])
		rest = list(map(int,row[1:]))
		data.append(np.array(rest))
data = np.asarray(data)
label = np.asarray(label)

#define kernel function
def mykernel(X, Y):
    return np.power(2*np.dot(X, Y.T) + 1,3)

#5-fold cross validation
print("Start CV")
sum1 = 0.0
kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
for train_index, test_index in kf.split(data,label):
	X_train, X_test = data[train_index], data[test_index]
	y_train, y_test = label[train_index], label[test_index]
	lsvc = SVC(kernel=mykernel).fit(X_train, y_train)
	y_pred = lsvc.predict(X_test)
	# print(sum((y_pred != y_test)*1.0/np.shape(X_test)[0]))
	sum1 += sum((y_pred != y_test)*1.0/np.shape(X_test)[0])
#print error rate
print(sum1/5.0)
