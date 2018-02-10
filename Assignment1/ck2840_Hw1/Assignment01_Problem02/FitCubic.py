#!/usr/bin/python3.5


def FitCubic():

    import numpy as np
    import pandas as pd
    import numpy.linalg as la
    from LinearRegressionSimulator import LinearRegressionSimulator
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages 
    """ UNI is ck2840 """

    std = 0.1
    theta = np.array([0,4,8,2])
    DataClass = LinearRegressionSimulator(theta,std)

    """ defining empirical test for test and training """
    rtest = np.ones((3, 11))
    rtrain = np.ones((3, 11))

    """ 3 cases are required for each N and M value """
    for run in range(0,3):
        if run == 0:
           N = 10
           M = 10
        elif run == 1:
           N = 100
           M = 10
        else:
           N = 10
           M = 100

        """ Generating data points """
        
        """ train data """
        dx = 1 / (N-1)
        Xtr = np.ones(N)
        for count in range(0,N):
            Xtr[count] = dx * count

        df1 = pd.DataFrame(Xtr)

        Ytr = DataClass.SimPoly(df1)

        """ test data """
        dx = 1 / (M-1)
        Xte = np.ones(M)
        for count in range(0,M):
            Xte[count] = dx * count

        df2 = pd.DataFrame(Xte)

        Yte = DataClass.SimPoly(df2)

        """ files to store variables """
        tr = np.ones(11)
        tt = np.ones(11)
        """ Calculating theta* """

        """ Creating Matrix X """
        for degree in range(0,11):

            Xm = np.ones([N, degree + 1])
            Xtm = np.ones([M, degree + 1])

            for j in range(0, degree+1):
                for i in range(0,N):
                    Xm[i][j]=pow(Xtr[i],j)

                for i in range(0,M):
                    Xtm[i][j]=pow(Xte[i], j)
                  
            """ Creating transpose of X """
            XT=Xm.transpose()

            """ calculating XTX """
            XTXm=np.matmul(XT, Xm)

            """ Calculating XTX inverse """ 
        
            if np.isfinite(la.cond(XTXm)):
                iXTXm=la.pinv(XTXm)
            else:
                print ("inverse of matrix XTX cannot be determined")

            """ Calculating product of inverse of (XTX) and XT """
            Xcum =np.matmul(iXTXm, XT)
           
            """ Calulating product of XM and Yi """
            thetat=np.matmul(Xcum, Ytr)

            """ Calculating empirical risk train data """
            tr[degree] = Rtrain=(1 / N) * pow(la.norm(np.subtract(Ytr,np.matmul(Xm, thetat))),2)
            rtrain[run][degree]=Rtrain

            """ Calculating empirical risk test data """
            tt[degree] = Rtest=(1 / M) * pow(la.norm(np.subtract(Yte, np.matmul(Xtm, thetat))),2)
            rtest[run][degree]=Rtest

            np.savetxt("ThetaStar["+str(run+1)+"]["+str(degree)+"].txt",thetat)

        """ Plotting files """   
        fname = "RiskPlot["+str(run+1)+"].pdf"
        pp = PdfPages(fname)
        plt.plot(rtrain[run], label = "train data")
        plt.plot(rtest[run], label = "test data")
        plt.ylabel("Emprical Risk")
        plt.xlabel("Degree")
        plt.legend()
        pp.savefig()
        plt.clf()
        pp.close()
     
        """ writing out the data files """
        np.savetxt("x.train.["+str(run+1)+"].txt", Xtr)
        np.savetxt("x.test.["+str(run+1)+"].txt", Xte)
        np.savetxt("y.train.["+str(run+1)+"].txt", Ytr)
        np.savetxt("y.test.["+str(run+1)+"].txt", Yte)
        np.savetxt("Risk.train.["+str(run+1)+"].txt", tr)
        np.savetxt("Risk.test.["+str(run+1)+"].txt", tt)

    return;
