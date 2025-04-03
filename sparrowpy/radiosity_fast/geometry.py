"""Geometry functions for the radiosity fast solver."""
try:
    import numba
    prange = numba.prange
except ImportError:
    numba = None
    prange = range
import numpy as np
import sparrowpy.radiosity_fast.visibility_helpers as vh


def get_scattering_data_receiver_index(
        pos_i:np.ndarray, pos_j:np.ndarray,
        receivers:np.ndarray, wall_id_i:np.ndarray,
        ):
    """Get scattering data depending on previous, current and next position.

    Parameters
    ----------
    pos_i : np.ndarray
        current position of shape (3)
    pos_j : np.ndarray
        next position of shape (3)
    receivers : np.ndarray
        receiver directions of all walls of shape (n_walls, n_receivers, 3)
    wall_id_i : np.ndarray
        current wall id to get write directional data

    Returns
    -------
    scattering_factor: float
        scattering factor from directivity

    """
    n_patches = pos_i.shape[0] if pos_i.ndim > 1 else 1
    receiver_idx = np.empty((n_patches), dtype=np.int64)

    for i in range(n_patches):
        difference_receiver = pos_i[i]-pos_j
        difference_receiver /= np.linalg.norm(
            difference_receiver)
        receiver_idx[i] = np.argmin(np.sum(
            (receivers[wall_id_i[i], :]-difference_receiver)**2, axis=-1),
            axis=-1)


    return receiver_idx


def get_scattering_data(
        pos_h:np.ndarray, pos_i:np.ndarray, pos_j:np.ndarray,
        sources:np.ndarray, receivers:np.ndarray, wall_id_i:np.ndarray,
        scattering:np.ndarray, scattering_index:np.ndarray):
    """Get scattering data depending on previous, current and next position.

    Parameters
    ----------
    pos_h : np.ndarray
        previous position of shape (3)
    pos_i : np.ndarray
        current position of shape (3)
    pos_j : np.ndarray
        next position of shape (3)
    sources : np.ndarray
        source directions of all walls of shape (n_walls, n_sources, 3)
    receivers : np.ndarray
        receiver directions of all walls of shape (n_walls, n_receivers, 3)
    wall_id_i : np.ndarray
        current wall id to get write directional data
    scattering : np.ndarray
        scattering data of shape (n_scattering, n_sources, n_receivers, n_bins)
    scattering_index : np.ndarray
        index of the scattering data of shape (n_walls)

    Returns
    -------
    scattering_factor: float
        scattering factor from directivity

    """
    difference_source = pos_h-pos_i
    difference_receiver = pos_i-pos_j

    difference_source /= np.linalg.norm(difference_source)
    difference_receiver /= np.linalg.norm(difference_receiver)
    source_idx = np.argmin(np.sum(
        (sources[wall_id_i, :, :]-difference_source)**2, axis=-1))
    receiver_idx = np.argmin(np.sum(
        (receivers[wall_id_i, :]-difference_receiver)**2, axis=-1))
    return scattering[scattering_index[wall_id_i],
        source_idx, receiver_idx, :]


def get_scattering_data_source(
        pos_h:np.ndarray, pos_i:np.ndarray,
        sources:np.ndarray, wall_id_i:np.ndarray,
        scattering:np.ndarray, scattering_index:np.ndarray):
    """Get scattering data depending on previous, current position.

    Parameters
    ----------
    pos_h : np.ndarray
        previous position of shape (3)
    pos_i : np.ndarray
        current position of shape (3)
    sources : np.ndarray
        source directions of all walls of shape (n_walls, n_sources, 3)
    wall_id_i : np.ndarray
        current wall id to get write directional data
    scattering : np.ndarray
        scattering data of shape (n_scattering, n_sources, n_receivers, n_bins)
    scattering_index : np.ndarray
        index of the scattering data of shape (n_walls)

    Returns
    -------
    scattering_factor: float
        scattering factor from directivity

    """
    difference_source = pos_h-pos_i
    difference_source /= np.linalg.norm(difference_source)
    source_idx = np.argmin(np.sum(
        (sources[wall_id_i, :, :]-difference_source)**2, axis=-1))
    return scattering[scattering_index[wall_id_i], source_idx]


def check_visibility(
        patches_center:np.ndarray,
        surf_normal:np.ndarray, surf_points:np.ndarray) -> np.ndarray:
    """Check the visibility between patches.

    Parameters
    ----------
    patches_center : np.ndarray
        center points of all patches of shape (n_patches, 3)
    surf_normal : np.ndarray
        normal vectors of all patches of shape (n_patches, 3)
    surf_points : np.ndarray
        boundary points of possible blocking surfaces (n_surfaces,)

    Returns
    -------
    visibility_matrix : np.ndarray
        boolean matrix of shape (n_patches, n_patches) with True if patches
        can see each other, otherwise false

    """
    n_patches = patches_center.shape[0]
    visibility_matrix = np.empty((n_patches, n_patches), dtype=np.bool_)
    visibility_matrix.fill(False)
    indexes = []
    for i_source in range(n_patches):
        for i_receiver in range(n_patches):
            if i_source < i_receiver:
                indexes.append((i_source, i_receiver))
                visibility_matrix[i_source,i_receiver]=True
    indexes = np.array(indexes)
    for i in prange(indexes.shape[0]):
        i_source = indexes[i, 0]
        i_receiver = indexes[i, 1]

        surfid=0
        while (visibility_matrix[i_source, i_receiver] and
                                surfid!=len(surf_normal)):

            visibility_matrix[i_source, i_receiver]= _basic_visibility(
                                                        patches_center[i_source],
                                                        patches_center[i_receiver],
                                                        surf_points[surfid],
                                                        surf_normal[surfid])

            surfid+=1

    return visibility_matrix


def _basic_visibility(vis_point: np.ndarray,
                     eval_point: np.ndarray,
                     surf_points: np.ndarray, surf_normal: np.ndarray,
                     eta=1e-6)->bool:
    """Return visibility of a point based on view point position.

    Parameters
    ----------
    vis_point: np.ndarray (3,)
        view point from which the visibility is being evaluated.

    eval_point: np.ndarray (3,)
        point being evaluated for visibility.

    surf_points: np.ndarray (N,3)
        boundary points of a possibly blocking surface.

    surf_normal: np.ndarray (3,)
        normal of possibly blocking surface.

    eta: float
        coplanarity check tolerance

    Returns
    -------
    is_visible: bool
        point visibility flag.
        (1 if points are visible to eachother, otherwise 0)

    """
    is_visible = True

    # if eval point is not coplanar with surf
    if np.abs(np.dot(surf_normal,eval_point-surf_points[0]))>eta:

        # check if projected point on surf
        pt = _project_to_plane(origin=vis_point, point=eval_point,
                            plane_pt=surf_points[0],
                            plane_normal=surf_normal,
                            check_normal=True)

        # if intersection point exists
        if pt is not None:
            # if plane is in front of eval point
            if (np.linalg.norm(eval_point-vis_point)>
                                np.linalg.norm(pt-vis_point)):
                # if point is inside surf polygon
                if _point_in_polygon(point3d=pt, polygon3d=surf_points,
                                    plane_normal=surf_normal):
                    is_visible = False

    # if both vis and eval point are coplanar
    elif (np.abs(np.dot(surf_normal,vis_point-surf_points[0]))<eta and
                    np.abs(np.dot(surf_normal,eval_point-surf_points[0]))<eta):
        is_visible = False

    return is_visible


def _create_patches(polygon_points:np.ndarray, max_size):
    """Create patches from a polygon."""
    size = np.empty(polygon_points.shape[1])
    for i in range(polygon_points.shape[1]):
        size[i] = polygon_points[:, i].max() - polygon_points[:, i].min()
    patch_nums = np.array([int(n) for n in size/max_size])
    real_size = size/patch_nums
    if patch_nums[2] == 0:
        x_idx = 0
        y_idx = 1
    if patch_nums[1] == 0:
        x_idx = 0
        y_idx = 2
    if patch_nums[0] == 0:
        x_idx = 1
        y_idx = 2

    x_min = np.min(polygon_points.T[x_idx])
    y_min = np.min(polygon_points.T[y_idx])

    n_patches = patch_nums[x_idx]*patch_nums[y_idx]
    patches_points = np.empty((n_patches, 4, 3))
    i = 0
    for i_x in range(patch_nums[x_idx]):
        for i_y in range(patch_nums[y_idx]):
            points = polygon_points.copy()
            points[0, x_idx] = x_min + i_x * real_size[x_idx]
            points[0, y_idx] = y_min + i_y * real_size[y_idx]
            points[1, x_idx] = x_min + (i_x+1) * real_size[x_idx]
            points[1, y_idx] = y_min + i_y * real_size[y_idx]
            points[3, x_idx] = x_min + i_x * real_size[x_idx]
            points[3, y_idx] = y_min + (i_y+1) * real_size[y_idx]
            points[2, x_idx] = x_min + (i_x+1) * real_size[x_idx]
            points[2, y_idx] = y_min + (i_y+1) * real_size[y_idx]
            patches_points[i] = points
            i += 1

    return patches_points


def _calculate_center(points):
    return np.sum(points, axis=-2) / points.shape[-2]


def _calculate_size(points):
    vec1 = points[..., 0, :]-points[..., 1, :]
    vec2 = points[..., 1, :]-points[..., 2, :]
    return np.abs(vec1-vec2)


def _calculate_area(points):
    area = np.zeros(points.shape[0])

    for i in prange(points.shape[0]):
        for tri in range(points.shape[1]-2):
            area[i] +=  .5 * np.linalg.norm(
                np.cross(
                    points[i, tri+1,:] - points[i, 0,:],
                    points[i, tri+2,:]-points[i, 0,:],
                    ),
                )

    return area


def process_patches(
        polygon_points_array: np.ndarray,
        walls_normal: np.ndarray,
        patch_size:float, n_walls:int):
    """Process the patches.

    Parameters
    ----------
    polygon_points_array : np.ndarray
        points of the polygon of shape (n_walls, 4, 3)
    walls_normal : np.ndarray
        wall normal of shape (n_walls, 3)
    patch_size : float
        maximal patch size in meters of shape (n_walls, 3).
    n_walls : int
        number of walls

    Returns
    -------
    patches_points : np.ndarray
        points of all patches of shape (n_patches, 4, 3)
    patches_normal : np.ndarray
        normal of all patches of shape (n_patches, 3)
    n_patches : int
        number of patches

    """
    n_patches = 0
    n_walls = polygon_points_array.shape[0]
    for i in range(n_walls):
        n_patches += total_number_of_patches(
            polygon_points_array[i, :, :], patch_size)
    patches_points = np.empty((n_patches, 4, 3))
    patch_to_wall_ids = np.empty((n_patches), dtype=np.int64)
    patches_per_wall = np.empty((n_walls), dtype=np.int64)

    for i in range(n_walls):
        polygon_points = polygon_points_array[i, :]
        patches_points_wall = _create_patches(
            polygon_points, patch_size)
        patches_per_wall[i] = patches_points_wall.shape[0]
        j_start = (np.sum(patches_per_wall[:i])) if i > 0 else 0
        j_end = np.sum(patches_per_wall[:i+1])
        patch_to_wall_ids[j_start:j_end] = i
        patches_points[j_start:j_end, :, :] = patches_points_wall
    n_patches = patches_points.shape[0]

    # calculate patch information
    patches_normal = walls_normal[patch_to_wall_ids, :]
    return (patches_points, patches_normal, n_patches, patch_to_wall_ids)


def total_number_of_patches(polygon_points:np.ndarray, max_size: float):
    """Calculate the total number of patches.

    Parameters
    ----------
    polygon_points : np.ndarray
        points of the polygon of shape (4, 3)
    max_size : float
        maximal patch size in meters

    Returns
    -------
    n_patches : int
        number of patches

    """
    size = np.empty(polygon_points.shape[1])
    for i in range(polygon_points.shape[1]):
        size[i] = polygon_points[:, i].max() - polygon_points[:, i].min()
    patch_nums = np.array([int(n) for n in size/max_size])
    if patch_nums[2] == 0:
        x_idx = 0
        y_idx = 1
    if patch_nums[1] == 0:
        x_idx = 0
        y_idx = 2
    if patch_nums[0] == 0:
        x_idx = 1
        y_idx = 2

    return patch_nums[x_idx]*patch_nums[y_idx]

def _calculate_normals(points: np.ndarray):
    """Calculate normals of plane defined by 3 or more input points.

    Parameters
    ----------
    points: np.ndarray (n_planes, n_points, 3)
        collection of points on common planes.

    Returns
    -------
    normals: np.ndarray (n_planes,3)
        collection of normal vectors to planes defined by point collection.
    """

    normals = np.empty((points.shape[0],3))

    for i in numba.prange(points.shape[0]):
        normals[i]=np.cross(points[i][1]-points[i][0],points[i][2]-points[i][0])
        normals[i]/=np.linalg.norm(normals[i])

    return normals

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

        b = _matrix_vector_product(np.linalg.inv(xmat), y)

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
        cc = _matrix_vector_product(matrix=rotation_matrix,vector=c)

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

        # if sample falls outside of triangular patch, it is "reflected" back inside
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


####################################################
# geometry

def _matrix_vector_product(matrix: np.ndarray,vector:np.ndarray)->np.ndarray:
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


def _rotation_matrix(n_in: np.ndarray, n_out=np.array([])):
    """Compute a rotation matrix from a given input and output directions.

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


def _sphere_tangent_vector(v0: np.ndarray, v1:np.ndarray) -> np.ndarray:
    """Compute a vector tangent to a spherical surface based on two points.

    The tangent vector is evaluated on point v0.
    The respective tangent arc on the sphere joins points v0 and v1.

    Parameters
    ----------
    v0 : numpy.ndarray(3,)
        point on which to calculate tangent vector

    v1 : numpy.ndarray(3,)
        point on sphere to which tangent vector "points"

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


def _project_to_plane(origin: np.ndarray, point: np.ndarray,
                     plane_pt: np.ndarray, plane_normal: np.ndarray,
                     epsilon=1e-6, check_normal=True):
    """Project point onto plane following direction defined by origin.

    Also applicable to lower or higher-dimensional spaces, not just 3D.

    Parameters
    ----------
    origin: np.ndarray(float) (n_dims,)
        point determining projection direction

    point: np.ndarray(float) (n_dims,)
        point to project

    plane_pt: np.ndarray(float) (n_dims,)
        reference point on projection plane

    plane_normal: np.ndarray(float) (n_dims,)
        normal of projection plane

    epsilon: float
        tolerance for plane orthogonality check

    check_normal: bool
        if True, discards projections on planes facing
        opposite direction from the point and origin.

    Returns
    -------
    int_point: np.ndarray(float) (n_dims,) or None
        intersection point.
        None if no intersection point is found

    """
    v = point-origin
    dotprod = np.dot(v,plane_normal)

    cond = dotprod < -epsilon

    if not check_normal:
        cond = abs(dotprod) > epsilon

    if cond:
        w = point-plane_pt
        fac = -np.divide(np.dot(plane_normal,w), dotprod)

        int_point = w + plane_pt + fac*v
    else:
        int_point = None

    return int_point


def _point_in_polygon(point3d: np.ndarray,
                     polygon3d: np.ndarray, plane_normal: np.ndarray,
                     eta=1e-6) -> bool:
    """Check if point is inside given polygon.

    Parameters
    ----------
    point3d: np.ndarray(float) (3,)
        point being evaluated.

    polygon3d: np.ndarray (N,3)
        polygon boundary points

    plane_normal: np.ndarray(float) (3,)
        normal of the polygon's plane

    eta: float
        tolerance for point inside of line check.

    Returns
    -------
    out: bool
        flags if point is inside polygon
        (True if inside, False if not)

    """
    # rotate all (coplanar) points to a horizontal plane
    # and remove z dimension for convenience
    rotmat = _rotation_matrix(n_in=plane_normal)

    pt = _matrix_vector_product(matrix=rotmat,vector=point3d)[0:point3d.shape[0]-1]
    poly = np.empty((polygon3d.shape[0],2))
    for i in prange(polygon3d.shape[0]):
        poly[i] = _matrix_vector_product(
            matrix=rotmat,vector=polygon3d[i])[0:point3d.shape[0]-1]


    # winding number algorithm
    count = 0
    for i in prange(poly.shape[0]):
        a1 = poly[(i+1)%poly.shape[0]]
        a0 = poly[i%poly.shape[0]]
        side = a1-a0

        # check if line from evaluation point in +x direction
        # intersects polygon side
        nl = np.array([-side[1],side[0]])/np.linalg.norm(side)
        b = _project_to_plane(origin=pt, point=pt+np.array([1.,0.]),
                             plane_pt=a1, plane_normal=nl,
                             check_normal=False)

        # check if intersection exists and if is inside side [a0,a1]
        if (b is not None) and b[0]>pt[0]:
            if abs(np.linalg.norm(b-a0)+np.linalg.norm(b-a1)
                                -np.linalg.norm(a1-a0)) <= eta:
                if np.dot(b-pt,nl)>0:
                    count+=1
                elif np.dot(b-pt,nl)<0:
                    count-=1

    if count != 0:
        out = True
    else:
        out = False

    return out


####################################################
# checks
def _coincidence_check(p0: np.ndarray, p1: np.ndarray, thres = 1e-3) -> bool:
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
    get_scattering_data_receiver_index = numba.njit()(
        get_scattering_data_receiver_index)
    total_number_of_patches = numba.njit()(total_number_of_patches)
    process_patches = numba.njit()(process_patches)
    _calculate_area = numba.njit()(_calculate_area)
    _calculate_center = numba.njit()(_calculate_center)
    _calculate_size = numba.njit()(_calculate_size)
    _create_patches = numba.njit()(_create_patches)
    get_scattering_data = numba.njit()(get_scattering_data)
    get_scattering_data_source = numba.njit()(get_scattering_data_source)
    check_visibility = numba.njit(parallel=True)(check_visibility)
    _calculate_normals = numba.njit()(_calculate_normals)
    _poly_estimation_Lagrange = numba.njit()(_poly_estimation_Lagrange)
    _poly_integration = numba.njit()(_poly_integration)
    polygon_area = numba.njit()(polygon_area)
    _surf_sample_random = numba.njit()(_surf_sample_random)
    _surf_sample_regulargrid = numba.njit()(_surf_sample_regulargrid)
    _sample_boundary_regular = numba.njit()(_sample_boundary_regular)
    _matrix_vector_product = numba.njit()(_matrix_vector_product)
    _rotation_matrix = numba.njit()(_rotation_matrix)
    _sphere_tangent_vector = numba.njit()(_sphere_tangent_vector)
    _area_under_curve = numba.njit()(_area_under_curve)
    _coincidence_check = numba.njit()(_coincidence_check)
    _basic_visibility = numba.njit()(_basic_visibility)
    _project_to_plane = numba.njit()(_project_to_plane)
    _point_in_polygon = numba.njit()(_point_in_polygon)

