"""Geometry functions for the form factor integration."""
try:
    import numba
    prange = numba.prange
except ImportError:
    numba = None
    prange = range
import numpy as np
import sparrowpy.geometry as geom


###################################################
# integration
################# 1D , polynomial
def _poly_estimation_Lagrange(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate Lagrange polynomial coefficients based on sample points.

    Computes coefficients of a polynomial curve passing through points (x,y)
    the order of the polynomial depends on the number of sample points
    input in the function. Uses the Lagrange method to estimate the polynomial.
        ex. a polynomial P estimated with 4 sample points:
            P4(x) = b[0]*x**3 + b[1]*x**2 + b[2]*x + b[3] = y

    Parameters
    ----------
    x: np.ndarray
        sample x-values
    y: np.ndarray
        sample y-values

    Returns
    -------
    b: np.ndarray
        polynomial coefficients

    """
    xmat = np.empty((len(x),len(x)))

    if np.abs(x[-1]-x[0])<1e-6:
        b = np.zeros(len(x))
    else:
        for i,xi in enumerate(x):
            for o in range(len(x)):
                xmat[i,len(x)-1-o] = xi**o

        b = geom._matrix_vector_product(np.linalg.inv(xmat), y)

    return b


def _poly_integration(c: np.ndarray, x: np.ndarray)-> float:
    """Integrate a polynomial curve.

    polynomial defined defined between x[0] and x[-1]
    with coefficients c
        ex. for a quadratic curve P2:
            P2(x) = c[0]*x**2 + c[1]*x + c[2]

    Parameters
    ----------
    c: np.ndarray
        polynomial coefficients
    x: np.ndarray
        sample points

    Returns
    -------
    out: float
        polynomial integral

    """
    out = 0

    for i in range(len(c)):
        out += c[i] * x[-1]**(len(c)-i) / (len(c)-i)
        out -= c[i] * x[0]**(len(c)-i) / (len(c)-i)

    return out

################# surface areas

def _area_under_curve(ps: np.ndarray, order=2) -> float:
    """Calculate the area under a polynomial curve.

    Curve sampled by a finite number of points ps on a common plane.

    Parameters
    ----------
    ps : np.ndarray
        sample points

    order : int
        polynomial order of the curve

    Returns
    -------
    area: float
        area under curve

    """
    # the order of the curve may be overwritten depending on the sample size
    order = min(order,len(ps)-1)

    # the vector between first and last sample (y==0) (new space's x axis)
    f  = ps[-1] - ps[0]

    rotation_matrix = np.array([[f[0],f[1]],[-f[1],f[0]]])/np.linalg.norm(f)

    x = np.zeros(order+1)
    y = np.zeros(order+1)

    for k in range(1,order+1):

        c = ps[k] - ps[0]  # translate point towards new origin

        # rotate point around origin to align with new axis
        cc = geom._matrix_vector_product(matrix=rotation_matrix,vector=c)

        x[k] = cc[0]
        y[k] = cc[1]


    coefs = _poly_estimation_Lagrange(x,y)
    area = _poly_integration(coefs,x) # area between curve and ps[-1] - ps[0]

    return area


####################################################
# sampling
################# surface
def _surf_sample_random(el: np.ndarray, npoints=100):
    """Randomly sample points on the surface of a patch.

    ! currently only supports triangular, rectangular, or parallelogram patches

    Parameters
    ----------
    el : geometry.Polygon object
        patch to sample

    npoints : int
        number of sample points to generate

    Returns
    -------
    ptlist: np.ndarray
        list of sample points in patch el

    """

    ptlist=np.zeros((npoints,3))

    u = el[1]-el[0]
    v = el[-1]-el[0]

    for i in range(npoints):
        s = np.random.uniform()
        t = np.random.uniform()

        inside = s+t <= 1

        # if sample falls outside of triangular patch,
        # it is "reflected" back inside
        if len(el)==3 and not inside:
            s = 1-s
            t = 1-t

        ptlist[i] = s*u + t*v + el[0]

    return ptlist


def _surf_sample_regulargrid(el: np.ndarray, npoints=10):
    """Sample points on the surface of a patch using a regular distribution.

    over the directions defined by the patches' sides

    ! currently only supports triangular, rectangular, or parallelogram patches
    ! may not return exact number of requested points
                -- depends on the divisibility of the patch

    Parameters
    ----------
    el : geometry.Polygon object
        patch to sample

    npoints : int
        number of sample points to generate

    Returns
    -------
    out: np.ndarray
        list of sample points in patch el

    """
    u = el[1]-el[0]
    v = el[-1]-el[0]

    if len(el)==3:
        a = 2
    else:
        a = 1

    npointsx = int( round(np.linalg.norm(u) / np.linalg.norm(v) *
                                                np.sqrt(a*npoints)) )
    npointsz = int( round(np.linalg.norm(v) / np.linalg.norm(u) *
                                                np.sqrt(a*npoints)) )

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

    for i,s in enumerate(tt):
        if len(el)==3:
            jj = i

        for t in tz[0:len(tz)-round(npointsz/npointsx*jj)]:

            inside = s+t <= 1-thres
            if not(len(el)==3 and not inside):
                ptlist.append(s*u + t*v + el[0])


    out = np.empty((len(ptlist), len(ptlist[0])))

    for i in prange(len(ptlist)):
        for j in prange(len(ptlist[0])):
            out[i][j] = ptlist[i][j]

    return out

################# boundary
def _sample_boundary_regular(el: np.ndarray, npoints=3):
    """Sample points on the boundary of a patch at fractional intervals.

    returns an array of points on the patch boundary (pts)
                                        and a connectivity array (conn)
    which stores a list of ordered indices of the points
    found on the same boundary segment.

    Parameters
    ----------
    el : geometry.Polygon object
        patch to sample

    npoints : int
        number of sample points per boundary segment (minimum 2)

    Returns
    -------
    pts: np.ndarray
        boundary sample points

    conn: np.ndarray(int)
        indices of pts corresponding to boundary segments
        (each row corresponds to the points in a single segment)

    """
    n_div = npoints - 1

    pts  = np.empty((len(el)*(npoints-1),len(el[0])))
    conn = np.empty((len(el),npoints), dtype=np.int8)

    for i in range(len(el)):

        conn[i][0]= (i*n_div)%(n_div*len(el))
        conn[i][-1]= (i*n_div+n_div)%(n_div*len(el))

        for ii in range(0,n_div):

            pts[i*n_div+ii,:]= (el[i] + ii*(el[(i+1)%len(el)]-el[i])/n_div)

            conn[i][ii]=(i*n_div+ii)%(n_div*len(el))


    return pts,conn.astype(np.int8)

if numba is not None:
    _poly_estimation_Lagrange = numba.njit()(_poly_estimation_Lagrange)
    _poly_integration = numba.njit()(_poly_integration)
    _surf_sample_random = numba.njit()(_surf_sample_random)
    _surf_sample_regulargrid = numba.njit()(_surf_sample_regulargrid)
    _sample_boundary_regular = numba.njit()(_sample_boundary_regular)
    _area_under_curve = numba.njit()(_area_under_curve)
