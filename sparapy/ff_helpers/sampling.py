import numpy as np
import numba
from sparapy.ff_helpers import geom


###################################################
# 1D polynomial integration

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


####################################################
#sampling


@numba.njit()
def sample_random(el: np.ndarray, npoints=100):
    """
    Randomly sample points on the surface of a patch using a uniform distribution
    
    ! currently only supports triangular, rectangular, or parallelogram patches

    Parameters
    ----------
    el : geometry.Polygon object
        patch to sample
            
    npoints : int
        number of sample points to generate
    """

    # TO DO: check that patch satisfies conditions for proper sampling
    # TO DO: if patch has >4 sides, subdivide into triangular patches and process independently ?

    ptlist=np.zeros((npoints,3)) 
    
    u = el[1]-el[0]
    v = el[-1]-el[0]

    for i in range(npoints):
        s = np.random.uniform()
        t = np.random.uniform()

        inside = s+t <= 1

        if len(el)==3 and not inside: # if sample falls outside of triangular patch, it is "reflected" inside 
            s = 1-s
            t = 1-t
        
        ptlist[i] = s*u + t*v + el[0]

    return ptlist

@numba.njit()
def sample_regular(el: np.ndarray, npoints=10):
    """
    Sample points on the surface of a patch using a regular distribution 
    over the directions defined by the patches' sides
    
    ! currently only supports triangular, rectangular, or parallelogram patches
    ! may not return exact number of requested points -- depends on the divisibility of the patch

    Parameters
    ----------
    el : geometry.Polygon object
        patch to sample
            
    npoints : int
        number of sample points to generate
    """
    
    # TO DO: check that patch satisfies conditions for proper sampling
    # TO DO: if patch has >4 sides, subdivide into triangular patches and process independently ?

    u = el[1]-el[0]
    v = el[-1]-el[0] 

    if len(el)==3:
        a = 2
    else:
        a = 1

    npointsx = int(round(np.linalg.norm(u)/np.linalg.norm(v)*np.sqrt(a*npoints)))
    npointsz = int(round(np.linalg.norm(v)/np.linalg.norm(u)*np.sqrt(a*npoints)))

    if npointsz==0:
        npointsz = 1
    if npointsx==0:
        npointsx = 1

    ptlist=[]

    tt = np.linspace(0,1-1/npointsx,npointsx)
    sstep =  1/(npointsx*2)
    tt += sstep

    tz = np.linspace(0,1-1/npointsz,npointsz)
    sstepz =  1/(npointsz*2)
    tz += sstepz

    thres = np.sqrt(sstepz**2+sstep**2)/2

    jj = 0

    # TO DO: find way to sample triangles more evenly 

    for i,s in enumerate(tt):
        if len(el)==3:
            jj = i

        for t in tz[0:len(tz)-round(npointsz/npointsx*jj)]: 

            inside = s+t <= 1-thres
            if not(len(el)==3 and not inside):
                ptlist.append(s*u + t*v + el[0])


    out = np.empty((len(ptlist), len(ptlist[0])))

    for i in numba.prange(len(ptlist)):
        for j in numba.prange(len(ptlist[0])):
            out[i][j] = ptlist[i][j]
    
    return out

@numba.njit()
def sample_border(el: np.ndarray, npoints=3):
    """
    Sample points on the boundary of a patch at fractional intervals of each side
    
    returns an array of points on the patch boundary (pts) 
            and a connectivity array (conn) which stores a list of ordered indices of the points (in spts) found on the same boundary segment

    Parameters
    ----------
    el : geometry.Polygon object
        patch to sample
            
    npoints : int
        number of sample points per boundary segment (minimum 2)
    """

    n_div = npoints - 1 # this function was written with a different logic in mind -- needs refactoring

    pts  = np.empty((len(el)*(npoints-1),len(el[0])))
    conn = np.empty((len(el),npoints), dtype=np.int8)

    for i in range(len(el)):

        conn[i][0]= (i*n_div)%(n_div*len(el))
        conn[i][-1]= (i*n_div+n_div)%(n_div*len(el))

        for ii in range(0,n_div):

            pts[i*n_div+ii,:]= (el[i] + ii*(el[(i+1)%len(el)]-el[i])/n_div)

            conn[i][ii]=(i*n_div+ii)%(n_div*len(el))


    return pts,conn.astype(np.int8)