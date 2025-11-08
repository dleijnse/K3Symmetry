import numpy as np
import matplotlib.pyplot as plt
import math as m


def f(pt):
    x = pt[0]
    y = pt[1]
    z = pt[2]
    return 1 - x**2 - y**2 - z**2 + x*y*z + x**4 + y**4 + z**4

fnp = np.vectorize(f)

delta = 0.05 # step size for computation
epsilon = 0.005 # we consider a point to be a zero of f if f(x,y,z) < epsilon
        # TODO: maybe let epsilon depend on the distance from the origin
bound = 1 # we consider x,y,z all between -bound and bound
steps = m.ceil(bound * 2 / delta)

X = np.arange(-bound, bound, delta)
Y = np.arange(-bound, bound, delta)
Z = np.arange(-bound, bound, delta)

xv, yv, zv = np.meshgrid(X, Y, Z, indexing='ij')

lst = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)

def cond(f, pt, e):
    return f(pt) < e

allowed_vals = np.array(list(filter(lambda pt : cond(f, pt, epsilon), lst)))

tr = np.transpose(allowed_vals)


ax = plt.figure().add_subplot(projection='3d')
ax.scatter(tr[0], tr[1], tr[2])

plt.show()
