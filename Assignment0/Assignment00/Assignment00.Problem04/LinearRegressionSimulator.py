import numpy as np
import pandas as pd

class  LinearRegressionSimulator:
    
     def __init__(self, Theta, StdDev):
         self.Theta = Theta
         self.StdDev = StdDev

     def SimData(self,XInput):
         nrow = len(XInput.index)
         flag = 0
         for x in range(0,nrow):
             mean = 0
             count = 0
             for y in np.nditer(self.Theta):
                 if count > 0: mean = mean+y*XInput.iat[x,count-1]
                 else:         mean = y
                 count = count + 1
             s = np.random.normal(mean, self.StdDev)
             if flag < 1: 
                   arr = np.array(s)
             else: 
                   arr = np.append(arr,s)
             flag = 2
         return arr
