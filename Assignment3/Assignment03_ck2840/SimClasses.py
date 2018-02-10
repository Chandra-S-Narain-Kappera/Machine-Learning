#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from numpy import linalg
from sklearn import  linear_model as lm
from sklearn import svm as sv
from cvxopt import matrix
from cvxopt import solvers
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle


class SimClasses:

    def GetData(self, N, D, Distance):

        # Setting up mean and covariance matrices

        mean_head = np.zeros(D)
        mean_tail = np.zeros(D)
        cov = np.zeros((D,D))
        Y = np.ones(N)
        X = []

        # For tails mean's first coordinate is distance D
        mean_tail[0] = Distance

        # Covariance matrix
        for i in range(0,D):
            for j in range(0,D):
                if i == j: cov[i][j] = 1

        # Drawing for Heads and Tails

        for i in range(0,N):
            if i%2 == 0:
                xarr = np.random.multivariate_normal(mean_head, cov)
                Y[i] = 1
            else:
                xarr = np.random.multivariate_normal(mean_tail, cov)
                Y[i] = -1
            if(i > 0):
                xarr_prev = np.vstack((xarr_prev, xarr))
            if i == 0:
                xarr_prev = xarr

        X = np.matrix(xarr_prev)

        return X,Y

class Classifier_A:

     # Logistic Regression
     def Classify(self, N, D, Distance):

         # Generate the data:
         cp = SimClasses()
         Xtr,Ytr = cp.GetData(N, D, Distance)

         # Fit the data:
         start = time.clock()
         lr = lm.LogisticRegression(C=1.0)
         lr.fit(Xtr,Ytr)
         end = time.clock() - start
         parameters = lr.coef_

         # Predict the data
         N = 100
         Xte,Yte = cp.GetData(N, D, Distance)
         Z = lr.predict(Xte)

         # Caclulate accuracy
         accuracy = (Yte.reshape(1,N) == Z)
         tmp = np.ones((1,N))
         accuracy = len(tmp[accuracy])
         return accuracy, end, parameters

class Classifier_B:

    # Perceptron Regression
    def Classify(self, N, D, Distance):

        # Generate the train data
        cp = SimClasses()
        Xtr, Ytr = cp.GetData(N, D, Distance)

        # Train the data
        pr = lm.Perceptron()
        start = time.clock()
        pr.fit(Xtr, Ytr)
        end = time.clock() - start


        # Test the data
        N = 100
        Xte, Yte = cp.GetData(N, D, Distance)
        Z = pr.predict(Xte)
        parameters = pr.coef_

        # Caclulate accuracy
        accuracy = (Yte.reshape(1,N) == Z)
        tmp = np.ones((1,N))
        accuracy = len(tmp[accuracy])
        return accuracy, end, parameters

class Classifier_C:

    # SVM Regression
    def Classify(self, N, D, Distance):

        # Generate the test data
        cp = SimClasses()
        Xtr, Ytr = cp.GetData(N, D, Distance)

        # Fit the data
        start = time.clock()
        svm = sv.SVC(kernel='linear')
        svm.fit(Xtr,Ytr)
        end = time.clock() - start
        parameters = svm.coef_

        # Predict the data
        N = 100
        Xte, Yte = cp.GetData(N, D, Distance)
        Z = svm.predict(Xte)

        # Caclulate accuracy
        accuracy = (Yte.reshape(1,N) == Z)
        tmp = np.ones((1,N))
        accuracy = len(tmp[accuracy])
        return accuracy, end, parameters

class Classifier_D:

    #SVM Regression Self Implemented:
    def Classify(self, N, D, Distance):

        #Getting training data:
        cp = SimClasses()
        X, Y = cp.GetData(N, D, Distance)

        # Fitting the data:
        start = time.clock()
        #Defining Matrices to be used as input for qp solver

        # Going to solve for alpha in dual form

        # penalty on slack
        soft_c = 1.0

        # Calculate Kernal type function (linear dot product)

        # Calculate other parameters to be passed onto CVXOPT solver
        X_tmp = X.getA()
        Y_tmp = Y[:, None]
        K = Y_tmp*X_tmp
        K = K.dot(K.T)
        P = matrix(K)
        q = matrix(-np.ones((N, 1)))
        tmp1 = np.diag(np.ones(N) * -1)
        tmp2 = np.identity(N)
        G = matrix(np.vstack((tmp1, tmp2)))
        t1_temp = np.zeros(N)
        t2_temp = np.ones(N)*soft_c
        h = matrix(np.hstack((t1_temp, t2_temp)))
        # Factoring in upper limit on C
        for i in range(N,N):
            h[i] = soft_c
        A = matrix(Y.reshape(1, -1))
        b = matrix(np.zeros(1))

        #Solve!!!
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)

        #Got alphas
        alphas = np.array(sol['x'])

        # get weights
        w = []
        sum = 0
        x_tmp = X.getA()
        for d in range(0,D):
            sum = 0
            for num in range(0,N):
                sum = sum + alphas[num]*Y[num]*x_tmp[num][d]
            w.append(sum)

        # get bias
        cond = np.logical_and(alphas > 1e-4, alphas < soft_c)
        cond = cond.reshape(-1)
        b = Y.reshape(N,1) - X.dot(w)
        b = b[cond]
        if len(b) is 0:
            bias = 0
        else:
            bias = np.array(b).mean()

        end = time.clock() - start
        parameters = np.array(w)

        # Predicting

        # Generate test data
        N = 100
        Xte, Yte = cp.GetData(N, D, Distance)

        # Test the points
        Xte = Xte.reshape(N,D)
        result = np.sign(Xte.dot(w)+bias)

        # Caclulate accuracy
        accuracy = (Yte.reshape(N,1) == result)
        tmp = np.ones((N,1))
        accuracy = len(tmp[accuracy])

        return accuracy, end, parameters


# Actual run:
def run_classifier(N, D, Distance, classifier):

    if classifier == "A":
        cp = Classifier_A()
    elif classifier == 'B':
        cp = Classifier_B()
    elif classifier == 'C':
        cp = Classifier_C()
    else:
        cp = Classifier_D()

    accuracy, train_time, parameters = cp.Classify(N, D, Distance)

    #print('Classifer: '+str(classifier)+ ' N = '+str(N)+' D = '+str(D)+' Distance = '+str(Distance)+'---> Accuracy = '+str(accuracy)+',Train_time = '+str(train_time)+' secs')
    return accuracy,format(train_time, '.4f'), parameters, parameters

def TestClassifiers():

    # Default values
    measurement = ['i', 'ii']
    method = ['A', 'B', 'C', 'D']
    item = ['a', 'b', 'c']
    val_a = [4, 5, 6, 7, 8]
    val_b = [25, 50, 75, 100, 200]
    val_c = [0.01, 0.1, 1, 10, 100]

    # Defaults for N, D, Distance
    N = 100
    D = 3
    Distance = 5
    Result_pkl = {}
    Parameters_pkl = {}
    for i in method:
        for j in item:
            if j == 'a':
                for k in val_a:
                    Result_pkl['i', i, j, k], Result_pkl['ii',i,j,k], Parameters_pkl['i',i,j,k], Parameters_pkl['ii',i,j,k] = run_classifier(N, k, Distance, i)
            if j == 'b':
                for k in val_b:
                    Result_pkl['i', i, j, k], Result_pkl['ii',i,j,k], Parameters_pkl['i',i,j,k], Parameters_pkl['ii',i,j,k] = run_classifier(k, D, Distance, i)

            if j == 'c':
                for k in val_c:
                    Result_pkl['i', i, j, k], Result_pkl['ii', i, j, k], Parameters_pkl['i',i,j,k], Parameters_pkl['ii',i,j,k] = run_classifier(N, D, k, i)

    # Data plots
    for i in measurement:
        if i == 'i': tag_1 = 'Accuracy'
        else:        tag_1 = 'Training time'
        for j in item:
            fname = 'Plot_'+i+'_'+j+'.pdf'
            pp = PdfPages(fname)
            if j == 'a':
                tag = 'Fixed: N ='+str(N)+', Distance ='+str(Distance)+', Variable: D'
                for k in method:
                    x = val_a
                    y = []
                    for l in val_a:
                        y.append(Result_pkl[i,k,j,l])
                    plt.plot(val_a, y, label = k)
                    plt.xticks(val_a)
            if j == 'c':
                tag = 'Fixed: N ='+ str(N) + ', D =' + str(D) + ', Variable: Distance'
                for k in method:
                    x = val_c
                    y = []
                    for l in val_c:
                        y.append(Result_pkl[i,k,j,l])
                    plt.plot(val_c, y, label=k)
            if j == 'b':
                tag = 'D =' + str(D) + ', Distance =' + str(Distance) + ', Variable: N'
                for k in method:
                    x = val_b
                    y = []
                    for l in val_b:
                        y.append(Result_pkl[i,k,j,l])
                    plt.plot(val_b, y, label=k)

            plt.ylabel(tag_1)
            plt.xlabel(tag)
            plt.legend()
            pp.savefig()
            plt.clf()
            pp.close()

    # Pickle file
    fname = "Results.pkl"
    with open(fname, 'wb') as handle:
        pickle.dump(Result_pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fname = "Parameters.pkl"
    with open(fname, 'wb') as handle:
        pickle.dump(Parameters_pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return

TestClassifiers()






