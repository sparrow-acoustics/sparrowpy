import numpy as np
import numba

@numba.njit(nopython=True)
def sample(p0,p1,nsamples=3):

    samples = np.empty([nsamples,len(p0)])

    step = np.empty([len(p0),1])

    for i in range(len(p0)):
        samples[:,i],step[i] = np.linspace(p0[i],p1[i],num=nsamples, retstep=True)

        
    return samples,step

@numba.njit(nopython=True)
def poly_estimation(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    xmat = np.empty(shape=[len(x),len(x)])

    if x[-1]-x[0]==0:
        return np.zeros([len(x,1)])
    else:
        for i,xi in enumerate(x):
            for o in range(len(x)):
                xmat[i,len(x)-1-o] = xi**o
        
        b = np.dot(np.linalg.inv(xmat), y)

        return b

@numba.njit(nopython=True)
def poly_integration(c: np.ndarray, x: np.ndarray)-> float:

    a0 = integ(c,x[0])
    a1 = integ(c,x[-1])

    # fig,a = plt.subplots()

    # a.plot(np.linspace(x[0], x[-1], 50),integ(c,np.linspace(x[0], x[-1], 50)))
    # a.fill_between(np.linspace(x[0], x[-1], 50), quad(c,np.linspace(x[0], x[-1], 50)), alpha=.5)
    # a.plot(x,quad(c,x),'r*')

    # plt.show()

    return a1-a0

@numba.njit(nopython=True)
def integ(b: np.ndarray, y: np.ndarray) -> float:

        out = 0

        for i in range(len(b)):

            out += b[i] * y**(len(b)-i) / (len(b)-i)

        return out
