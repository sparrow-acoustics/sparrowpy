"""Module for form factor calculation."""
import numpy as np
import sparrowpy.radiosity_fast.universal_ff.ffhelpers as helpers
try:
    import numba
    prange = numba.prange
except ImportError:
    numba = None
    prange = range


#/////////////////////////////////////////////////////////////////////////////////////#
#######################################################################################
### patch-to-patch
def calc_form_factor(
        receiving_pts: np.ndarray, receiving_normal: np.ndarray,
        source_pts: np.ndarray, source_normal: np.ndarray) -> float:
    """
    Calculate form factor depending on the characteristics of the input
    patches.

    Parameters
    ----------
    receiving_pts: np.ndarray
        receiver patch vertex coordinates (n_vertices,3)

    receiving_normal: np.ndarray
        receiver patch normal (3,)

    source_pts: np.ndarray
        source patch vertex coordinates (n_vertices,3)

    source_normal: np.ndarray
        source patch normal (3,)

    Returns
    -------
    out: float
        form factor

    """

    if helpers.coincidence_check(
            receiving_pts, source_pts):
        out = nusselt_integration(
            patch_i=source_pts, patch_i_normal=source_normal,
            patch_j=receiving_pts, patch_j_normal=receiving_normal,
            nsamples=64)
    else:
        out = stokes_integration(
            patch_i=source_pts, patch_j=receiving_pts,
            source_area=helpers.polygon_area(source_pts), approx_order=4)

    return out


#######################################################################################
### Stokes integration
def stokes_ffunction(p0:np.ndarray, p1: np.ndarray) -> float:
    """
    Returns the form function value for the stokes form factor integration
    method.

    Parameters
    ----------
    p0: np.ndarray
        a point in space (3,)
        in the stokes integration of the form factor, a point on a patch's
        boundary

    p1: np.ndarray
        a point in space (3,)
        in the stokes integration of the form factor, a point on a different
        patch's boundary

    Returns
    -------
    result: float
        form function for points p1, p0
    """

    n = np.linalg.norm(p1-p0)

    result = np.log(n)

    return result


def load_stokes_entries(
        i_bpoints: np.ndarray, j_bpoints: np.ndarray) -> np.ndarray:
    """
    Calculates all the stokes form function values between two patches
    and stores them in a matrix.

    Parameters
    ----------
    i_bpoints: np.ndarray
        list of points in patch i boundary (n_boundary_points_i , 3)

    j_bpoints: np.ndarray
        list of points in patch j boundary (n_boundary_points_j , 3)

    Returns
    -------
    form_mat: np.ndarray
        matrix of form function values
        (n_boundary_points_i , n_boundary_points_j)

    """

    form_mat = np.zeros((i_bpoints.shape[0] , j_bpoints.shape[0]))

    for i in prange(i_bpoints.shape[0]):
        for j in prange(j_bpoints.shape[0]):
            form_mat[i,j] = stokes_ffunction(i_bpoints[i],j_bpoints[j])

    return form_mat


def stokes_integration(
        patch_i: np.ndarray, patch_j: np.ndarray, source_area: float,
        approx_order=4) -> float:
    """
    Calculate an estimation of the form factor between two patches
    by computationally integrating a modified form function over the two
    patch boundaries.
    The modified form function follows Stokes' theorem.

    The modified form function integral is calculated using a polynomial
    approximation based on sampled values.

    Parameters
    ----------
    patch_i : np.ndarray
        vertex coordinates of patch i (n_vertices, 3)

    patch_j : np.ndarray
        vertex coordinates of patch j (n_vertices, 3)

    source_area: float
        area of the source patch

    approx_order: int
        polynomial order of the form function integration estimation

    Returns
    -------
    form_factor: np.ndarray
        form factor between patch_i and patch_j using Stokes' theorem
        for integration
    """

    i_bpoints, i_conn = helpers.sample_border(patch_i, npoints=approx_order+1)
    j_bpoints, j_conn = helpers.sample_border(patch_j, npoints=approx_order+1)

    subsecj = np.zeros((j_conn.shape[1]))
    subseci = np.zeros((i_conn.shape[1]))
    form_mat = np.zeros((i_bpoints.shape[0],j_bpoints.shape[0]))

    # first compute and store form function sample values
    form_mat = load_stokes_entries(i_bpoints, j_bpoints)

    # double polynomial integration (per dimension (x,y,z))
    outer_integral = 0
    inner_integral = np.zeros((len(i_bpoints),len(j_bpoints[0])))

    # for each dimension
    for dim in range(len(j_bpoints[0])):
        # integrate form function over each point on patch i boundary

        # for each point in patch i boundary
        for i in range(len(i_bpoints)):
            # for each segment segj in patch j boundary
            for segj in j_conn:
                xj = j_bpoints[segj][:,dim]

                if xj[-1]-xj[0]!=0:
                    for k in range(len(segj)):
                        subsecj[k] = form_mat[i][segj[k]]
                    # compute polynomial coefficients of approx form function
                    # over boundary segment xj
                    quadfactors = helpers.poly_estimation(x=xj, y=subsecj)
                    # analytical integration of the approx polynomia
                    inner_integral[i][dim] += helpers.poly_integration(
                        c=quadfactors,x=xj)


        # integrate previously computed integral over each boundary segment
        # of patch i

        # for each segment segi in patch i boundary
        for segi in i_conn:

            xi = i_bpoints[segi][:,dim]

            if xi[-1]-xi[0]!=0:
                for k in range(len(segi)):
                    subseci[k] = inner_integral[segi[k]][dim]
                quadfactors = helpers.poly_estimation(x=xi, y=subseci)
                outer_integral += helpers.poly_integration(c=quadfactors,x=xi)

    return np.abs(outer_integral/(2*np.pi*source_area))


###############################################################################
### Nusselt analog integration

def nusselt_analog(
        surf_origin, surf_normal, patch_points, patch_normal) -> float:
    """
    Implementation of the nusselt analog.
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

    """

    # 3 points per boundary segment (for quadratic approximation)
    boundary_points, connectivity = helpers.sample_border(
        patch_points, npoints=3)

    hand = np.sign(np.dot(np.cross(
        patch_points[1]-patch_points[0], patch_points[2]-patch_points[1],
        ), patch_normal) )

    curved_area = 0

    sphPts = np.empty_like( boundary_points )
    projPts = np.empty_like( boundary_points )
    plnPts = np.empty( shape=(len(boundary_points),2) )

    for ii in prange(len(boundary_points)):
        # patch j points projected on the hemisphere
        sphPts[ii] = (boundary_points[ii]-surf_origin)/np.linalg.norm(
            boundary_points[ii]-surf_origin)

    rotmat = helpers.rotation_matrix(n_in=surf_normal)

    for ii in prange(len(sphPts)):
        # points on the hemisphere projected onto
        plnPts[ii,:] = helpers.inner(matrix=rotmat,vector=sphPts[ii])[:-1]
        projPts[ii,:-1] = plnPts[ii,:]
        projPts[ii,-1] = 0.


    big_poly = helpers.polygon_area(projPts[0::2])

    segmt=np.empty_like(connectivity[0])

    leftseg=np.empty((3,2))
    rightseg=np.empty((3,2))

    for jj in prange(connectivity.shape[0]):

        segmt = connectivity[jj]

        if np.linalg.norm(np.cross(
                projPts[segmt[-1]],projPts[segmt[0]])) > 1e-20:

            # if the points on the segment span less than 90 degrees relative
            # to the origin
            if np.dot( plnPts[segmt[-1]], plnPts[segmt[0]] ) >= 0:
                curved_area += helpers.area_under_curve(plnPts[segmt],order=2)

            # if points span over 90º, additional sampling is required
            else:
                mpoint = sphPts[segmt[0]] + (
                    sphPts[segmt[-1]] - sphPts[segmt[0]]) / 2
                # midpoint on the arc projected on the hemisphere
                marc = mpoint/np.linalg.norm(mpoint)
                a = sphPts[segmt[0]] + (marc - sphPts[segmt[0]]) / 2
                b = marc + (sphPts[segmt[-1]] - marc) / 2

                mpoint = helpers.inner(matrix=rotmat,vector=mpoint)[:-1]
                marc = helpers.inner(matrix=rotmat,vector=marc)[:-1]
                a = a/np.linalg.norm(a)
                a = helpers.inner(matrix=rotmat,vector=a)[:-1]

                b = b/np.linalg.norm(b)
                b = helpers.inner(matrix=rotmat,vector=b)[:-1]

                linArea = np.linalg.norm(
                    plnPts[segmt[-1]] - plnPts[segmt[0]]) * np.linalg.norm(
                        mpoint-marc)/2

                leftseg[0] = plnPts[segmt[0]]
                leftseg[1] = a
                leftseg[2] = marc

                rightseg[0] = marc
                rightseg[1] = b
                rightseg[2] = plnPts[segmt[-1]]


                left =  helpers.area_under_curve(leftseg, order=2)
                right = helpers.area_under_curve(rightseg, order=2)
                curved_area += (linArea * np.sign(left) + left + right)

    return big_poly + hand*curved_area


def nusselt_integration(
        patch_i: np.ndarray, patch_j: np.ndarray, patch_i_normal: np.ndarray,
        patch_j_normal: np.ndarray, nsamples=2, random=False) -> float:
    """
    Estimates the form factor by integrating the differential form factor
    (Nusselt analogue)
    over the surface of the source patch.

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

    nsamples: int
        number of receiver surface samples for integration

    random: bool
            determines the distribution of the samples on patch_i surface
            if True, the samples are randomly distributed in a uniform way
            if False, a regular sampling of the surface is performed
    """

    if random:
        p0_array = helpers.sample_random(patch_i,nsamples)
    else:
        p0_array = helpers.sample_regular(patch_i,nsamples)

    out = 0

    for i in prange(p0_array.shape[0]):
        out += nusselt_analog(
            surf_origin=p0_array[i], surf_normal=patch_i_normal,
            patch_points=patch_j, patch_normal=patch_j_normal)

    out *= helpers.polygon_area(patch_i)/(
        np.pi * len(p0_array) * helpers.polygon_area(patch_j))

    return out


#/////////////////////////////////////////////////////////////////////////////////////#
#######################################################################################
### point-to-patch and patch-to-point
def pt_solution(point: np.ndarray, patch_points: np.ndarray, mode='source'):
    """
    Calculates the "form factor" between a point (source or receiver) and
    a patch using a modified version of the Nusselt analogue, transformed
    for a -point- source, rather than differential surface element.

    Parameters
    ----------
    point: np.ndarray
        source or receiver point

    patch_points: np.ndarray
        vertex coordinates of the patch

    mode: string
        determines if point is acting as a source ('source')
        or as a receiver ('receiver')
    """

    if mode == 'receiver':
        source_area = helpers.polygon_area(patch_points)
    elif mode == 'source':
        source_area = 4

    npoints = len(patch_points)

    interior_angle_sum = 0

    patch_onsphere = np.zeros_like(patch_points)

    for i in range(npoints):
        patch_onsphere[i]= (patch_points[i]-point) / np.linalg.norm(
            patch_points[i]-point)

    for i in range(npoints):

        v0 = helpers.calculate_tangent_vector(
            patch_onsphere[i], patch_onsphere[(i-1)%npoints])
        v1 = helpers.calculate_tangent_vector(
            patch_onsphere[i], patch_onsphere[(i+1)%npoints])

        interior_angle_sum += np.arccos(np.dot(v0,v1))

    factor = interior_angle_sum - (len(patch_points)-2)*np.pi

    return factor / (np.pi*source_area)


if numba is not None:
    calc_form_factor = numba.njit()(calc_form_factor)
    stokes_ffunction = numba.njit()(stokes_ffunction)
    pt_solution= numba.njit(parallel=True)(pt_solution)
    nusselt_integration = numba.njit(parallel=True)(nusselt_integration)
    nusselt_analog = numba.njit(parallel=False)(nusselt_analog)
    load_stokes_entries = numba.njit(parallel=True)(load_stokes_entries)
    stokes_integration = numba.njit(parallel=False)(stokes_integration)
