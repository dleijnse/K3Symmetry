import numpy as np
import matplotlib.pyplot as plt
import math as m

from skimage import measure

def f(x, y, z):
    return 0.4 + 0.7 * (x**2 + y**2 + z**2) + 1.6 * x*y*z - 1.3 *  (x**4 + y**4 + z**4)

def fpt(pt):
    x = pt[0]
    y = pt[1]
    z = pt[2]
    return f(x, y, z)

fnp = np.vectorize(f)

def plot_scatter():
    delta = 0.05 # step size for computation
    epsilon = 0.001 # we consider a point to be a zero of f if f(x,y,z) < epsilon
            # TODO: maybe let epsilon depend on the distance from the origin
    bound = 3 # we consider x,y,z all between -bound and bound
    steps = m.ceil(bound * 2 / delta)

    X = np.arange(-bound, bound, delta)
    Y = np.arange(-bound, bound, delta)
    Z = np.arange(-bound, bound, delta)

    xv, yv, zv = np.meshgrid(X, Y, Z, indexing='ij')

    lst = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)

    def cond(fpt, pt, e):
        return f(pt) < e

    allowed_vals = np.array(list(filter(lambda pt : cond(fpt, pt, epsilon), lst)))

    tr = np.transpose(allowed_vals)


    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(tr[0], tr[1], tr[2])

    plt.show()

def plot_marching():
    delta = 0.05
    bound = 1
    steps = m.ceil(bound * 2 / delta)
    xs = np.linspace(-bound, bound, steps)
    X, Y, Z = np.meshgrid(xs, xs, xs)
    F = f(X, Y, Z)
    verts, faces, normals, values = measure.marching_cubes(F, 0, spacing=[np.diff(xs)[0]]*3)
    verts -= 3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='jet', lw=0)
    plt.show()


plot_marching()
