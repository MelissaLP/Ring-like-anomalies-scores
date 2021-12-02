import numba
from numba import vectorize
import numpy as np


@numba.njit(fastmath=True)
def temperature(inner, v):
    # "inner" refers to pixels inside the anulus.
    # We get a list of temperatures according to pixels' ID
    return v[inner]


@numba.njit(fastmath=True)
def minus(query, query_in):
    # The inverse intersection between query (outer circle),
    # and query_in (inner circle) yields the pixels of the anulus
    return np.array(list(set(query) - set(query_in)))


@numba.njit(fastmath=True)
def A_positive(F):
    # A function for the upper tail
    return -10000*(np.log(1-np.power(F, 10000)).sum())/len(F)


@numba.njit(fastmath=True)
def A_negative(F):
    # A function for the lower tail
    return -10000*np.log(1-np.power(1-F, 10000)).sum()/len(F)


@vectorize(["float64(float64, float64, float64,float64, float64, float64)"])
def scalar_product(x1, y1, z1, x2, y2, z2):
    return x1*x2+y1*y2+z1*z2


@vectorize(["float64(float64, float64, float64,float64, float64, float64)"])
def cross(x1, y1, z1, x2, y2, z2):
    return np.sqrt((y1*z2 - z1*y2)**2+(z1*x2 - x1*z2)**2+(x1*y2 - y1*x2)**2)


@numba.njit(fastmath=True, parallel=True)
def angular_distance(vec, pixels):
    phi = np.zeros(len(pixels))
    for i in range(len(pixels)):
        vec2 = np.array([pixels[i][0], pixels[i][1], pixels[i][2]])
        phi[i] = np.arctan2(cross(vec[0], vec[1], vec[2], vec2[0], vec2[1], vec2[2]),
                            scalar_product(vec[0], vec[1], vec[2], vec2[0], vec2[1], vec2[2]))
    return phi


@numba.njit(fastmath=True, parallel=True)
def linearity(distance, ring):
    grad = ((len(ring)*np.sum(distance*ring)-np.sum(distance)*np.sum(ring))/(len(ring)*np.sum(distance**2)-np.sum(distance)**2))
    r_coeff = (len(ring)*np.sum(distance*ring)-np.sum(distance)*np.sum(ring))/np.sqrt((len(ring)*np.sum(distance**2)-np.sum(distance)**2)*(len(ring)*np.sum(ring**2)-np.sum(ring)**2))
    return grad, r_coeff
