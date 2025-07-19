"""Methods and helpers for the form factor integration."""
try:
    import numba
    prange = numba.prange
except ImportError:
    numba = None
    prange = range
import numpy as np
import sparrowpy.geometry as geom


def load_stokes_entries(
    i_bpoints: np.ndarray, j_bpoints: np.ndarray) -> np.ndarray:
    """Load all the stokes form function values between two patches.

    Parameters
    ----------
    i_bpoints: np.ndarray
        list of points in patch i boundary (n_boundary_points_i , 3)

    j_bpoints: np.ndarray
        list of points in patch j boundary (n_boundary_points_j , 3)

    Returns
    -------
    form_mat: np.ndarray
        f function value matrix (n_boundary_points_i , n_boundary_points_j)

    """
    form_mat = np.zeros((len(i_bpoints) , len(j_bpoints)))

    for i in prange(i_bpoints.shape[0]):
        for j in prange(j_bpoints.shape[0]):
            form_mat[i][j] = np.log(np.linalg.norm(i_bpoints[i]-j_bpoints[j]))

    return form_mat

def stokes_integration(
    patch_i: np.ndarray, patch_j: np.ndarray, patch_i_area: float) -> float:
    """Calculate an estimation of the form factor between two patches.

    Computationally integrates a modified form function over
    the boundaries of both patches.
    The modified form function follows Stokes' theorem.

    The modified form function integral is calculated using a
    polynomial approximation based on sampled values.

    Parameters
    ----------
    patch_i : np.ndarray
        vertex coordinates of patch i (n_vertices, 3)

    patch_i_area: float
       area of patch i

    patch_j : np.ndarray
        vertex coordinates of patch j (n_vertices, 3)

    source_area: float
        area of the source patch

    approx_order: int
        polynomial order of the form function integration estimation

    Returns
    -------
    float
    form factor between two patches

    """
    i_bpoints, i_conn = _sample_boundary_regular(patch_i,
                                                         npoints=5)
    j_bpoints, j_conn = _sample_boundary_regular(patch_j,
                                                         npoints=5)

    subsecj = np.zeros((j_conn.shape[1]))
    subseci = np.zeros((i_conn.shape[1]))
    form_mat = np.zeros((i_bpoints.shape[0],j_bpoints.shape[0]))

    # first compute and store form function sample values
    form_mat = load_stokes_entries(i_bpoints, j_bpoints)

    # double polynomial integration (per dimension (x,y,z))
    outer_integral = 0
    inner_integral = np.zeros((len(i_bpoints),len(j_bpoints[0])))

    for dim in range(len(j_bpoints[0])): # for each dimension
        # integrate form function over each point on patch i boundary

        for i in range(len(i_bpoints)):   # for each point in patch i boundary
            for segj in j_conn:     # for each segment segj in patch j boundary

                xj = j_bpoints[segj][:,dim]

                if np.abs(xj[-1]-xj[0])>1e-3:
                    for k in range(len(segj)):
                        subsecj[k] = form_mat[i][segj[k]]

                    inner_integral[i][dim]+=_newton_cotes_4th(xj,subsecj)



        # integrate previously computed integral over patch i
        for segi in i_conn:   # for each segment segi in patch i boundary

            xi = i_bpoints[segi][:,dim]

            if np.abs(xi[-1]-xi[0])>1e-3:
                for k in range(len(segi)):
                    subseci[k] = inner_integral[segi[k]][dim]
                outer_integral+= _newton_cotes_4th(xi,subseci)

    return np.abs(outer_integral/(2*np.pi*patch_i_area))

def nusselt_analog(surf_origin, surf_normal,
                   patch_points, patch_normal) -> float:
    """Calculate the Nusselt analog for a single point.

    Projects a given receiver patch onto a hemisphere centered around a point
    on a source patch surface.
    The hemispherical projection is then projected onto the source patch plane.
    The area of this projection relative to the unit circle area is the
    differential form factor between the two patches.

    Parameters
    ----------
    surf_origin : np.ndarray
        point on source patch for differential form factor evaluation (3,)
        (global origin)

    surf_normal : np.ndarray
        normal of source patch (3,)

    patch_points : np.ndarray
        vertex coordinates of the receiver patch (n_vertices, 3)

    patch_normal: np.ndarray
        normal of receiver patch (3,)

    Returns
    -------
    Nusselt analog factor
    (differential form factor)

    """
    boundary_points, connectivity = _sample_boundary_regular(
                                                            patch_points,
                                                            npoints=3)

    hand = np.sign(np.dot(
                np.cross(patch_points[1]-patch_points[0],
                         patch_points[2]-patch_points[1]), patch_normal) )

    curved_area = 0

    sphPts = np.empty_like( boundary_points )
    projPts = np.empty_like( boundary_points )
    plnPts = np.empty( shape=(len(boundary_points),2) )

    for ii in prange(len(boundary_points)):
        # patch j points projected on the hemisphere
        sphPts[ii] = ( (boundary_points[ii]-surf_origin) /
                        np.linalg.norm(boundary_points[ii]-surf_origin) )

    rotmat = geom._rotation_matrix(n_in=surf_normal)

    for ii in prange(len(sphPts)):
        # points on the hemisphere projected onto patch plane
        plnPts[ii,:] = geom._matrix_vector_product(matrix=rotmat,
                                                      vector=sphPts[ii])[:-1]
        projPts[ii,:-1] = plnPts[ii,:]
        projPts[ii,-1] = 0.


    big_poly = geom._polygon_area(projPts[0::2])

    segmt=np.empty_like(connectivity[0])

    leftseg=np.empty((3,2))
    rightseg=np.empty((3,2))

    for jj in prange(connectivity.shape[0]):

        segmt = connectivity[jj]

        if (np.linalg.norm(np.cross(projPts[segmt[-1]],projPts[segmt[0]]))
                                                                    > 1e-6):

            # if the points on the segment span less than 90 degrees
            if np.dot( plnPts[segmt[-1]], plnPts[segmt[0]] ) >= 1e-6:
                curved_area += _area_under_curve(plnPts[segmt],order=2)

            # if points span over 90º, additional sampling is required
            else:
                mpoint = ( sphPts[segmt[0]] +
                          (sphPts[segmt[-1]] - sphPts[segmt[0]]) / 2 )

                # midpoint on the arc projected on the hemisphere
                marc = mpoint/np.linalg.norm(mpoint)
                a = sphPts[segmt[0]] + (marc - sphPts[segmt[0]]) / 2
                b = marc + (sphPts[segmt[-1]] - marc) / 2

                mpoint = geom._matrix_vector_product(matrix=rotmat,
                                                        vector=mpoint)[:-1]
                marc = geom._matrix_vector_product(matrix=rotmat,
                                                      vector=marc)[:-1]
                a = a/np.linalg.norm(a)
                a = geom._matrix_vector_product(matrix=rotmat,vector=a)[:-1]

                b = b/np.linalg.norm(b)
                b = geom._matrix_vector_product(matrix=rotmat,vector=b)[:-1]

                linArea = (np.linalg.norm(plnPts[segmt[-1]]-plnPts[segmt[0]])
                                              * np.linalg.norm(mpoint-marc)/2)

                leftseg[0] = plnPts[segmt[0]]
                leftseg[1] = a
                leftseg[2] = marc

                rightseg[0] = marc
                rightseg[1] = b
                rightseg[2] = plnPts[segmt[-1]]


                left =  _area_under_curve(leftseg, order=2)
                right = _area_under_curve(rightseg, order=2)
                curved_area += (linArea * np.sign(left) + left + right)

    return big_poly + hand*curved_area

def nusselt_integration(patch_i: np.ndarray, patch_j: np.ndarray,
                        patch_i_normal: np.ndarray, patch_j_normal: np.ndarray,
                        nsamples=2, random=False) -> float:
    """Estimate the form factor based on the Nusselt analogue.

    Integrates the differential form factor (Nusselt analogue output)
    over the surface of the source patch

    Parameters
    ----------
    patch_i: np.ndarray
        vertex coordinates of the source patch

    patch_j: np.ndarray
        vertex coordinates of the receiver patch

    patch_i_normal: np.ndarray
        source patch normal (3,)

    patch_j_normal: np.ndarray
        receiver patch normal (3,)

    patch_i_area: float
        source patch area

    patch_j_area: float
        receiver patch area

    nsamples: int
        number of receiver surface samples for integration

    random: bool
        determines the distribution of the samples on patch_i surface
        if True, the samples are randomly distributed in a uniform way
        if False, a regular sampling of the surface is performed

    Returns
    -------
    out: float
        form factor between patches i and j

    """
    if random:
        p0_array = _surf_sample_random(patch_i,nsamples)
    else:
        p0_array = _surf_sample_regulargrid(patch_i,nsamples)

    out = 0

    for i in prange(p0_array.shape[0]):
        out += nusselt_analog( surf_origin=p0_array[i],
                               surf_normal=patch_i_normal,
                               patch_points=patch_j,
                               patch_normal=patch_j_normal )

    out *= 1 / ( np.pi * len(p0_array) )

    return out


#/////////////////////////////////////////////////////////////////////////////////////#
#######################################################################################
### point-to-patch and patch-to-point
def pt_solution(point: np.ndarray, patch_points: np.ndarray, mode='source'):
    """Calculate the geometric factor between a point and a patch.

    applies a modified version of the Nusselt analogue,
    transformed for a -point- source rather than differential surface element.

    Parameters
    ----------
    point: np.ndarray
        source or receiver point

    patch_points: np.ndarray
        vertex coordinates of the patch

    mode: string
        determines if point is acting as a source ('source')
        or as a receiver ('receiver')

    Returns
    -------
    geometric factor

    """
    if mode == 'receiver':
        source_area = geom._polygon_area(patch_points)
    elif mode == 'source':
        source_area = 4

    npoints = len(patch_points)

    interior_angle_sum = 0

    patch_onsphere = np.zeros_like(patch_points)

    for i in range(npoints):
        patch_onsphere[i]= ( (patch_points[i]-point) /
                              np.linalg.norm(patch_points[i]-point) )

    for i in range(npoints):

        v0 = geom._sphere_tangent_vector(patch_onsphere[i],
                                              patch_onsphere[(i-1)%npoints])
        v1 = geom._sphere_tangent_vector(patch_onsphere[i],
                                              patch_onsphere[(i+1)%npoints])

        interior_angle_sum += np.arccos(np.dot(v0,v1))

    factor = interior_angle_sum - (len(patch_points)-2)*np.pi

    return factor / (np.pi*source_area)

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

def _newton_cotes_4th(x: np.ndarray,y):
    """Integrate 1D function after Boole's rule.

    Returns the approximate integral of a given function f(x)
    given 5 equally spaced samples (x,y).
    x samples the function input, while y samples the respective f(x) values.

    The integral is numerically estimated via the closed 4th-order
    Newton-Cotes formula (aka Boole's Rule).
    This formula *requires* 5 equidistant sample input.

    Parameters
    ----------
    x: np.ndarray(float)
        x-coordinate of the input samples.
    y: np.ndarray(float)
        f(x) of the input samples.
    """

    h=x[1]-x[0]
    return 2*h/45 *(7*y[0] +
                    32*y[1] +
                    12*y[2] +
                    32*y[3] +
                    7*y[4])

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
    pt_solution = numba.njit(parallel=True)(pt_solution)
    nusselt_integration = numba.njit(parallel=False)(nusselt_integration)
    stokes_integration = numba.njit(parallel=False)(stokes_integration)
    nusselt_analog = numba.njit(parallel=False)(nusselt_analog)
    load_stokes_entries = numba.njit(parallel=True)(load_stokes_entries)
    _poly_estimation_Lagrange = numba.njit()(_poly_estimation_Lagrange)
    _poly_integration = numba.njit()(_poly_integration)
    _surf_sample_random = numba.njit()(_surf_sample_random)
    _surf_sample_regulargrid = numba.njit()(_surf_sample_regulargrid)
    _sample_boundary_regular = numba.njit()(_sample_boundary_regular)
    _area_under_curve = numba.njit()(_area_under_curve)
    _newton_cotes_4th = numba.njit()(_newton_cotes_4th)
