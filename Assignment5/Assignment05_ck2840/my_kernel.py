import numpy as np

def euclidean_dist(x, y):
    """
    Returns matrix of pairwise, squared Euclidean distances
    """
    norms_1 = (x ** 2).sum(axis=1)
    norms_2 = (y ** 2).sum(axis=1)
    y = np.transpose(y)
    prod = np.dot(x, y)
    prod = 2*prod
    norms_1 = norms_1.reshape(-1,1)
    sum = norms_1 + norms_2
    sum = sum -  prod
    abs_mat = np.abs(sum)
    return abs_mat

def prbf_kernel(x,y):
    """
    Combination of Polynomial + RBF Kernel
    """
    gamma = 0.05
    dists_sq = euclidean_dist(x, y)
    z = (10+np.exp(-gamma * dists_sq))
    return z
