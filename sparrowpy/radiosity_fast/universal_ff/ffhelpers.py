"""universal form factor helper methods."""
import numpy as np
try:
    import numba
    prange = numba.prange
except ImportError:
    numba = None
    prange = range


###################################################
# integration
################# 1D , polynomial
@numba.njit()
def poly_estimation_Lagrange(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Estimate polynomial coefficients based on sample points.

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

        b = inner(np.linalg.inv(xmat), y)

    return b


def poly_integration(c: np.ndarray, x: np.ndarray)-> float:
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
def polygon_area(pts: np.ndarray) -> float:
    """Calculate the area of a convex n-sided polygon.

    Parameters
    ----------
    pts: np.ndarray
        list of 3D points which define the vertices of the polygon

    Returns
    -------
    area: float
        area of polygon defined by pts

    """
    area = 0

    for tri in range(pts.shape[0]-2):
        area  +=  .5 * np.linalg.norm(
                            np.cross(pts[tri+1] - pts[0], pts[tri+2]-pts[0]) )

    return area


def area_under_curve(ps: np.ndarray, order=2) -> float:
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
        cc = inner(matrix=rotation_matrix,vector=c)

        x[k] = cc[0]
        y[k] = cc[1]


    coefs = poly_estimation_Lagrange(x,y)
    area = poly_integration(coefs,x) # area between curve and ps[-1] - ps[0]

    return area

@numba.njit()
def pascal_array(order: int):
    """
    Compute Pascal's triangle as a np.array up to given order.

    Parameters
    ----------
    order: int
        number of rows of Pascal's triangle to compute.

    Returns
    -------
    triangle: np.array(order,order)
        Pascal's triangle (lower triangular matrix).

    """

    triangle = np.zeros((order,order))
    triangle[:,0] = 1

    for i in range(1,order):
        triangle[i,1:i+1] = triangle[i-1,1:i+1] + triangle[i-1,0:i]

    return triangle

####################################################
# sampling
################# surface
def sample_random(el: np.ndarray, npoints=100):
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
    # TO DO: check that patch satisfies conditions for proper sampling
    # TO DO: if patch has >4 sides,
    #           subdivide into triangular patches and process independently ?

    ptlist=np.zeros((npoints,3))

    u = el[1]-el[0]
    v = el[-1]-el[0]

    for i in range(npoints):
        s = np.random.uniform()
        t = np.random.uniform()

        inside = s+t <= 1

        # if sample falls outside of triangular patch, it is "reflected" inside
        if len(el)==3 and not inside:
            s = 1-s
            t = 1-t

        ptlist[i] = s*u + t*v + el[0]

    return ptlist


def sample_regular(el: np.ndarray, npoints=10):
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
def sample_border(el: np.ndarray, npoints=3):
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


####################################################
# geometry

def inner(matrix: np.ndarray,vector:np.ndarray)->np.ndarray:
    """Compute the inner product between a matrix and a vector to please njit.

    Parameters
    ----------
    matrix : numpy.ndarray(n,n)
        input matrix

    vector : numpy.ndarray(n,)
        vector

    Returns
    -------
    out: np.ndarray(n,)
        matrix*vector inner product

    """
    out = np.empty(matrix.shape[0])

    for i in prange(matrix.shape[0]):
        out[i] = np.dot(matrix[i],vector)

    return out


def rotation_matrix(n_in: np.ndarray, n_out=np.array([])):
    """Compute a rotation matrix from a given input and output directions.

    TO DO: expand to N-D arrays

    Parameters
    ----------
    n_in : numpy.ndarray(3,)
        input vector

    n_out : numpy.ndarray(3,)
        direction to which n_in is to be rotated

    Returns
    -------
    matrix: np.ndarray
        rotation matrix

    """
    if n_out.shape[0] == 0:
        n_out = np.zeros_like(n_in)
        n_out[-1] = 1.
    else:
        n_out = n_out

    #check if all the vector entries coincide
    counter = int(0)

    for i in prange(n_in.shape[0]):
        if n_in[i] == n_out[i]:
            counter+=1
        else:
            counter=counter

    # if input vector is the same as output return identity matrix
    if counter == n_in.shape[0]:

        matrix = np.eye( len(n_in) , dtype=np.float64)

    else:

        a = n_in / np.linalg.norm(n_in)
        a = np.reshape(a, len(n_in) )

        b = n_out / np.linalg.norm(n_out)
        b = np.reshape(b, len(n_in) )

        c = np.dot(a,b)

        if c!=-1:
            v = np.cross(a,b)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]],
                             [v[2], 0, -v[0]],
                             [-v[1], v[0], 0]])

            matrix =  ( np.eye( len(n_in) ) +
                       kmat +
                       kmat.dot(kmat) * ((1 - c) / (s ** 2)) )

        else: # in case the in and out vectors have symmetrical directions
            matrix = np.array([[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]])

    return matrix


def calculate_tangent_vector(v0: np.ndarray, v1:np.ndarray) -> np.ndarray:
    """Compute a vector tangent to a spherical surface based on two points.

    The tangent vector has is evaluated on point v0
    and its tangent arc on the sphere joins points v0 and v1

    Parameters
    ----------
    v0 : numpy.ndarray(3,)
        point on which to calculate tangent vector

    v1 : numpy.ndarray(3,)
        point on sphere to which tangent vector poinst

    Returns
    -------
    vout: np.ndarray
        vector tangent to spherical surface

    """
    if np.abs(np.dot(v0,v1))>1e-10:
        vout = (v1-v0)-np.dot((v1-v0),v0)/np.dot(v0,v0)*v0
        vout /= np.linalg.norm(vout)

    else:
        vout = v1/np.linalg.norm(v1)

    return vout


####################################################
# checks
@numba.njit()
def coincidence_check(p0: np.ndarray, p1: np.ndarray, thres = 1e-3) -> bool:
    """Flag true if two patches have any common points.

    Parameters
    ----------
    p0 : numpy.ndarray(# vertices, 3)
        patch

    p1 : numpy.ndarray(# vertices, 3)
        another patch

    thres: float
        threshold value for distance between points

    Returns
    -------
    flag: bool
        flag of coincident points on both patches

    """
    flag = False

    for i in numba.prange(p0.shape[0]):
        for j in numba.prange(p1.shape[0]):
            if np.linalg.norm(p0[i]-p1[j])<thres:
                flag=True
                break


    return flag


if numba is not None:
    poly_estimation_Lagrange = numba.njit()(poly_estimation_Lagrange)
    poly_integration = numba.njit()(poly_integration)
    polygon_area = numba.njit()(polygon_area)
    sample_random = numba.njit()(sample_random)
    sample_regular = numba.njit()(sample_regular)
    sample_border = numba.njit()(sample_border)
    inner = numba.njit()(inner)
    rotation_matrix = numba.njit()(rotation_matrix)
    calculate_tangent_vector = numba.njit()(calculate_tangent_vector)
    area_under_curve = numba.njit()(area_under_curve)
    coincidence_check = numba.njit()(coincidence_check)
