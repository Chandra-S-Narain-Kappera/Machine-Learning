#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


class LinearRegressionSimulator(object):
    def __init__(self, Theta, std):
        """
        Inputs:
            Theta - array of coefficients (nonempty 1xD+1 numpy array)
            std   - standard deviation (float)
        """

        assert len(Theta) != 0

        self.Theta = Theta
        self.std = std

    def SimData(self, XInput):
        """
        Input:
            XInput - (NxD pandas dataframe)

        Returns:
            outarray - (N-dim vector)
        """
        N, D = XInput.shape

        assert D + 1 == len(self.Theta)

        self.means = self.Theta[0] + np.matmul(XInput, self.Theta[1:])
        outarray = self.std * np.random.randn(N) + self.means

        return outarray

    def SimPoly(self, XInput):
        """
        Input:
            XInput - (NxD pandas dataframe)

        Returns:
            outarray - (N-dim vector)
        """
        D = len(XInput.index)

        self.means = self.Theta[0]

        nrow = len(XInput.index)
        flag = 0
        for x in range(0, nrow):
            mean = 0
            count = 0
            for y in np.nditer(self.Theta):
                if count > 0:
                    mean = mean + y * pow(XInput.iat[x, 0], count)
                else:
                    mean = y
                count = count + 1
                s = np.random.normal(mean, self.std)
            if flag < 1:
                outarray = np.array(s)
            else:
                outarray = np.append(outarray, s)
            flag = 2

        return outarray

