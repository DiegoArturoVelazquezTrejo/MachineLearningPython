import numpy as np
from numpy import linalg
import cvxpt
import cvxpt.solvers

def linear_kernel(x1,x2):
    return np.dot(x1,x2)
def polynomial_kernel(x,y,p=3):
    return (1 + np.dot(x,y)) **p

def gaussian_kernel(x,y,sigma=5.0):
    return np.exp(-linalg.norm(x-y) ** 2 / (2 * (sigma ** 2)))

#tut 32
