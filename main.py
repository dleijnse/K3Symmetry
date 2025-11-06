import numpy as np
import matplotlib
import math as m


def f(x, y, z):
    return 1 - x**2 - y**2 - z**2 + x*y*z + x**4 + y**4 + z**4

delta = 2 # step size for computation
epsilon = 0.00001 # we consider a point to be a zero of f if f(x,y,z) < epsilon
        # TODO: maybe let epsilon depend on the distance from the origin
bound = 5 # we consider x,y,z all between -bound and bound
steps = m.ceil(bound * 2 / delta)

X = np.arange(-bound, bound, delta)
Y = np.arange(-bound, bound, delta)
Z = np.arange(-bound, bound, delta)
grid = np.meshgrid(X, Y, Z)

# TODO: flatten this meshgrid, it seems 2-dimensional now

print(grid)
