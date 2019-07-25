import numpy as np
from matplotlib import pyplot as plt

#inner product = dot product in numpy
#A similarity function which takes the similarities between two inputs and two outputs, using the inner product.
#Use a kernel to transform our feature space.

#Non linearly separable feature sets



'''
#SV / #Samples, if this division is more than a hundred, edata might not be
linearly separable. Try a different kernel.


k(x,x¨) = z . z'

z = function(x)
z' = function(x')

dot product between z and z'   -> that´s a kernel

y = w . kernelx + b

A inner product produces a scalar value!
The kernel is the inner product of the z space.


feature set   -> [x1, x2]  -> second order polynomial

X = [x1,x2] Z = [1,x1,x2, x1**2 , x2**2 , x1*x2 ]

K(x,x')    Z' = [1 , x1', x2', x1**-2 , x2**-2 , x1', x2' ]
K(x,x') = (1 + x . x' )**p

(1 + x1x1' + ... + xnxn')**p

exp(x) = e ** x


SLACK must be greater or equal than zero.
SLACK ->  introduce slack variable yi(xiw+b) >= 1 - slack
'''
