import numpy as np
import numba
from sparapy.ff_helpers import geom

@numba.njit()
def sample(p0,p1,nsamples=3):

    samples = np.empty([nsamples,len(p0)])

    step = np.empty([len(p0),1])

    for i in range(len(p0)):
        samples[:,i],step[i] = np.linspace(p0[i],p1[i],num=nsamples, retstep=True)

        
    return samples,step

@numba.njit()
def poly_estimation(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    xmat = np.empty((len(x),len(x)))

    if x[-1]-x[0]==0:
        b = np.zeros(len(x))
    else:
        for i,xi in enumerate(x):
            for o in range(len(x)):
                xmat[i,len(x)-1-o] = xi**o
        
        b = geom.inner(np.linalg.inv(xmat), y)

    return b

@numba.njit()
def poly_integration(c: np.ndarray, x: np.ndarray)-> float:

    out = 0

    for j in [-1,0]:
        for i in range(len(c)):
            out -= -1**j * c[i] * x[j]**(len(c)-i) / (len(c)-i)

    return out

@numba.njit()
def integ(b: np.ndarray, y: np.ndarray) -> float:

    out = 0

    for i in range(len(b)):
        out += b[i] * y**(len(b)-i) / (len(b)-i)

    return out


