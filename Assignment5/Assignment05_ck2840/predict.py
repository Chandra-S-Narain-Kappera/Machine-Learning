####################################################
# COMS 4771 Machine Learning Assignment 5 Problem 2
# Name: Chandra Shankaradithya Narain Kappera
# UNI:CK2840 
####################################################

import pickle
from my_kernel import *

def predictSVM(X_test):
	"""This function takes a dataset and predict the letter for each data point.

	Parameters
	----------
	dataset: M X 128 numpy array
		A dataset represented by numpy-array

	Returns
	-------
	M x 1 numpy array
		Returns a numpy array of letter that each data point represents
	"""
	fp = open("clf.pkl",'rb')  
	# load the object from the file into var clf
	clf = pickle.load(fp)
	z = clf.predict(X_test)
	fp.close()
	return z
