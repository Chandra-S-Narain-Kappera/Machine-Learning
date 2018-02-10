#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

def RegularizedFitPoly():

    import numpy as np
    import pandas as pd
    import numpy.linalg as la
    from LinearRegressionSimulator import LinearRegressionSimulator
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import pickle

    # UNI is ck2840

    std = 0.1
    theta = np.array([0, 4, 8, 2])
    DataClass = LinearRegressionSimulator(theta, std)

    # Setting values for test and training data
    N = 10
    M = 100

    # creating dictionaries to stores values for pickle file
    data_dict   = dict()
    theta_dict  = dict()
    risktr_dict = dict()
    riskte_dict = dict()

    # Generating data points

    # Train data
    dx = 2 / (N - 1)
    Xtr = np.ones(N)
    for count in range(0, N):
        Xtr[count] = -1 + dx * count

    df1 = pd.DataFrame(Xtr)

    Ytr = DataClass.SimPoly(df1)

    # Test data
    dx = 2 / (M - 1)
    Xte = np.ones(M)
    for count in range(0, M):
        Xte[count] = -1 + dx * count

    df2 = pd.DataFrame(Xte)

    Yte = DataClass.SimPoly(df2)

    # files to store training and test variables

    tr = np.ones(5)
    tt = np.ones(5)

    # Calculating theta*

    # Creating Matrix X

    Xm = np.ones([N, 7])
    Xtm = np.ones([M, 7])


    for i in range(0, M):

        if i < N :
            Xm[i][0] = 1
            Xm[i][1] = Xtr[i]
            Xm[i][2] = 0.5*(3*pow(Xtr[i], 2) - 1)
            Xm[i][3] = 0.5*(5*pow(Xtr[i], 3) - 3*Xtr[i])
            Xm[i][4] = 0.25*(35*pow(Xtr[i], 4) - 30*pow(Xtr[i], 2) + 3)
            Xm[i][5] = 0.25*(63*pow(Xtr[i], 5) - 70*pow(Xtr[i], 3) + 15*Xtr[i])
            Xm[i][6] = 0.0625*(231*pow(Xtr[i], 6) - 315*pow(Xtr[i], 4) + 105*pow(Xtr[i], 2) - 5)


        Xtm[i][0] = 1
        Xtm[i][1] = Xte[i]
        Xtm[i][2] = 0.5*(3*pow(Xte[i], 2) - 1)
        Xtm[i][3] = 0.5*(5*pow(Xte[i], 3) - 3*Xte[i])
        Xtm[i][4] = 0.25*(35*pow(Xte[i], 4) - 30*pow(Xte[i], 2) + 3)
        Xtm[i][5] = 0.25*(63*pow(Xte[i], 5) - 70*pow(Xte[i], 3) + 15*Xte[i])
        Xtm[i][6] = 0.0625*(231*pow(Xte[i], 6) - 315*pow(Xte[i], 4) + 105*pow(Xte[i], 2) - 5)


    # Creating transpose of X
    XT = Xm.transpose()

    # Calculate XTX
    XTXm = np.matmul(XT, Xm)

    for param in range (0, 5):

        if   param == 0: lmbda = 0
        elif param == 1: lmbda = 0.001
        elif param == 2: lmbda = 0.1
        elif param == 3: lmbda = 10
        elif param == 4: lmbda = 100

        # Calculating N*lmbda*I
        I = np.ones([7, 7])
        for i in range (0, 7):
            for j in range (0, 7):
                if i == j: I[i][j] = N*lmbda
                else     : I[i][j] = 0


        # Calculating the inverse
        iXTXm = la.pinv(XTXm+I)

        # Calculating product of inverse of (XTX) and XT
        Xcum = np.matmul(iXTXm, XT)

        # Calulating product of XM and Yi
        thetat = np.matmul(Xcum, Ytr)

        #Adding thetat to dictionary
        theta_dict[lmbda] = thetat

        # Calculating empirical risk train data
        reg_parm = lmbda*0.5*pow(la.norm(thetat), 2)
        tr[param] = Rtrain = (1 / N) * pow(la.norm(np.subtract(Ytr, np.matmul(Xm, thetat))), 2) + reg_parm

        # Calculating empirical risk test data
        tt[param] = Rtest = (1 / M) * pow(la.norm(np.subtract(Yte, np.matmul(Xtm, thetat))), 2) + reg_parm

        # Adding risk values to dictionary
        riskte_dict[lmbda] = Rtrain
        risktr_dict[lmbda] = Rtest

        # Generating data to get plots
        ytemp = np.matmul(Xtm, thetat)

        plt.plot(Xte, ytemp, label= "$\lambda$="+str(lmbda))

    # Plotting the data files
    fname = "ApproximationPlot.pdf"
    pp = PdfPages(fname)

    # Training Data
    plt.plot(Xtr, Ytr, 'ro', label="Train Data")

    # Original Data plots
    for i in range(0, M):
        ytemp[i] = 2 * pow(Xte[i], 3) + 8 * pow(Xte[i], 2) + 4 * Xte[i]
    plt.plot(Xte, ytemp , label="Original Data")

    plt.ylabel("Y")
    plt.xlabel("X")
    plt.legend()
    pp.savefig()
    plt.clf()
    pp.close()

    # Adding data to the dictionary
    data_dict["xtrain"] = Xtr
    data_dict["xtest"]  = Xtr
    data_dict["ytrain"] = Ytr
    data_dict["ytest"]  = Yte
    data_dict["ThetaStar"] = theta_dict
    data_dict["RiskTest"] =  risktr_dict
    data_dict["RiskTrain"] = riskte_dict

    #Creating pickle file
    fname = "problem1.pkl"

    with open(fname, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return
