import numpy as np
import sparapy.ffhelpers as helpers
import numba

#/////////////////////////////////////////////////////////////////////////////////////#
#######################################################################################
### patch-to-patch
@numba.njit()
def calc_form_factor(receiving_pts: np.ndarray, receiving_normal: np.ndarray, source_pts: np.ndarray, source_normal: np.ndarray) -> float:

    if helpers.coincidence_check(receiving_pts, source_pts):
        out = nusselt_integration(patch_i=source_pts, patch_i_normal=source_normal, patch_j=receiving_pts, patch_j_normal=receiving_normal, nsamples=64)
    else:
        out = stokes_integration(patch_i=source_pts, patch_j=receiving_pts, source_area=helpers.polygon_area(source_pts),  approx_order=4)
    
    return out

#######################################################################################
### Stokes integration 
@numba.njit()
def stokes_ffunction(p0:np.ndarray, p1: np.ndarray) -> float:

    n = np.linalg.norm(p1-p0)

    result = np.log(n)
    return result

@numba.njit(parallel=True)
def load_stokes_entries(i_bpoints: np.ndarray, j_bpoints: np.ndarray) -> np.ndarray:

    form_mat = np.empty((len(j_bpoints) , len(i_bpoints)))

    for j in numba.prange(i_bpoints.shape[0]):
        for i in numba.prange(j_bpoints.shape[0]):
            form_mat[i][j] = stokes_ffunction(j_bpoints[i],i_bpoints[j])

    return form_mat

@numba.njit(parallel=True)
def stokes_integration(patch_i: np.ndarray, patch_j: np.ndarray, source_area: float, approx_order=4) -> float:
    """
    calculate an estimation of the form factor between two patches 
    by computationally integrating a modified form function over the two patch boundaries.
    The modified form function follows Stokes' theorem.

    The modified form function integral is calculated using a polynomial approximation based on sampled values.
    
    Parameters
    ----------
    patch_i : geometry.Polygon
        radiance emitting patch

    patch_j : geometry.Polygon
        radiance receiving patch

    approx_order: int
        determines the order of the polynomial integration order. 
        also determines the number of samples in each patch's boundary.

    """

    i_bpoints, i_conn = helpers.sample_border(patch_i, npoints=approx_order+1)
    j_bpoints, j_conn = helpers.sample_border(patch_j, npoints=approx_order+1)

    subsec = np.empty((j_conn.shape[1]))
    form_mat = np.empty((i_bpoints.shape[0],j_bpoints.shape[0]))

    # if coincidence_check(i_bpoints,j_bpoints): 
    #     return float('nan')

    # first compute and store form function sample values
    form_mat = load_stokes_entries(i_bpoints, j_bpoints)


    # double polynomial integration (per dimension (x,y,z))
    outer_integral = 0
    inner_integral = np.zeros((len(i_bpoints),len(j_bpoints[0])))

    for dim in numba.prange(len(j_bpoints[0])):                                # for each dimension
        # integrate form function over each point on patch i boundary

        for i in numba.prange(len(i_bpoints)):                                 # for each point in patch i boundary
            for segj in j_conn:                                         # for each segment segj in patch j boundary
                
                xj = j_bpoints[segj][:,dim]          

                if xj[-1]-xj[0]!=0:
                    for k in range(len(segj)):
                        subsec[k] = form_mat[segj[k]][dim]
                    quadfactors = helpers.poly_estimation(xj, subsec) # compute polynomial coefficients of approx form function over boundary segment xj
                    inner_integral[i][dim] += helpers.poly_integration(quadfactors,xj)                          # analytical integration of the approx polynomial


        # integrate previously computed integral over each boundary segment of patch i

        for segi in i_conn:                     # for each segment segi in patch i boundary

            xi = i_bpoints[segi][:,dim]

            if xi[-1]-xi[0]!=0:
                for k in range(len(segi)):
                    subsec[k] = inner_integral[segi[k]][dim]
                quadfactors = helpers.poly_estimation(xi,subsec ) 
                outer_integral += helpers.poly_integration(quadfactors,xi)


    return np.abs(outer_integral/(2*np.pi*source_area))


#######################################################################################
### Nusselt analog integration
@numba.njit(parallel=False)
def nusselt_analog(surf_origin, surf_normal, patch_points, patch_normal) -> float:

    boundary_points, connectivity = helpers.sample_border(patch_points, npoints=3) # 3 points per boundary segment (for quadratic approximation)

    hand = np.sign(np.dot(np.cross( patch_points[1]-patch_points[0] , patch_points[2]-patch_points[1] ), patch_normal) )

    curved_area = 0
        
    sphPts = np.empty_like( boundary_points )
    projPts = np.empty_like( boundary_points )
    plnPts = np.empty( shape=(len(boundary_points),2) )

    for ii in numba.prange(len(boundary_points)):
        sphPts[ii] = (boundary_points[ii]-surf_origin)/np.linalg.norm(boundary_points[ii]-surf_origin) # patch j points projected on the hemisphere

    rotmat = helpers.rotation_matrix(n_in=surf_normal)

    for ii in numba.prange(len(sphPts)):
        plnPts[ii,:] = helpers.inner(rotmat,sphPts[ii])[:-1] # points on the hemisphere projected onto 
        projPts[ii,:-1] = plnPts[ii,:]
        projPts[ii,-1] = 0.


    big_poly = helpers.polygon_area(projPts[0::2])

    segmt=np.empty_like(connectivity[0])

    leftseg=np.empty((3,boundary_points.shape[-1]))
    rightseg=np.empty((3,boundary_points.shape[-1]))

    for jj in numba.prange(connectivity.shape[0]):

        segmt = connectivity[jj]

        if np.linalg.norm(np.cross(projPts[segmt[-1]],projPts[segmt[0]])) > 1e-12:

            if np.dot( plnPts[segmt[-1]], plnPts[segmt[0]] ) >= 0:                    # if the points on the segment span less than 90 degrees relative to the origin
                curved_area += helpers.area_under_curve(plnPts[segmt],order=2)

            else:                                                                       # if points span over 90º, additional sampling is required
                mpoint = sphPts[segmt[0]] + (sphPts[segmt[-1]] - sphPts[segmt[0]]) / 2
                marc = mpoint/np.linalg.norm(mpoint) # midpoint on the arc projected on the hemisphere
                a = sphPts[segmt[0]] + (marc - sphPts[segmt[0]]) / 2
                b = marc + (sphPts[segmt[-1]] - marc) / 2

                mpoint = helpers.inner(helpers.rotation_matrix(surf_normal),mpoint)[:-1]
                marc = helpers.inner(helpers.rotation_matrix(surf_normal),marc)[:-1]
                a = a/np.linalg.norm(a)
                a = helpers.inner(helpers.rotation_matrix(surf_normal),a)[:-1]

                b = b/np.linalg.norm(b)
                b = helpers.inner(helpers.rotation_matrix(surf_normal),b)[:-1]

                linArea = np.linalg.norm(plnPts[segmt[-1]] - plnPts[segmt[0]]) * np.linalg.norm(mpoint-marc)/2


                leftseg[0] = plnPts[segmt[0]]
                leftseg[1] = a
                leftseg[2] = marc

                rightseg[0] = marc
                rightseg[1] = b
                rightseg[2] = plnPts[segmt[-1]]


                left =  helpers.area_under_curve(leftseg, order=2)
                right = helpers.area_under_curve(rightseg, order=2)
                curved_area += hand*(linArea * np.sign(left) + left + right)

    return big_poly + curved_area

@numba.njit(parallel=True)
def nusselt_integration(patch_i: np.ndarray, patch_j: np.ndarray, patch_i_normal: np.ndarray, patch_j_normal: np.ndarray, nsamples=2, random=False) -> float:
    """
    Estimates the form factor by integrating the Nusselt analogue values (emitting patch) 
    over the surface of the receiving patch

    Parameters
    ----------
    patch_i: geometry.Polygon
            emitting patch

    patch_j: geometry.Polygon
            receiving patch

    nsamples: int
            number of surface samples for integration

    random: bool
            determines the distribution of the samples on patch_i surface 
            if True, the samples are randomly distributed in a uniform way
            if False, a regular sampling of the surface is performed
    """
    
    if random:
        p0_array = helpers.sample_random(patch_j,nsamples)
    else:
        p0_array = helpers.sample_regular(patch_j,nsamples)

    out = 0

    for i in numba.prange(p0_array.shape[0]):
        out += nusselt_analog(surf_origin=p0_array[i], surf_normal=patch_j_normal, patch_points=patch_i, patch_normal=patch_i_normal)

    out /= (np.pi * len(p0_array))

    return out 


#/////////////////////////////////////////////////////////////////////////////////////#
#######################################################################################
### point-to-patch and patch-to-point
def pt_solution(point: np.ndarray, patch_points: np.ndarray, mode='source'):
    """

    """
    if mode == 'receiver':
        source_area = helpers.polygon_area(patch_points)
    elif mode == 'source':
        source_area = 4

    npoints = len(patch_points)

    interior_angle_sum = 0

    patch_onsphere = np.divide( (patch_points-point) , np.repeat(np.linalg.norm(patch_points-point, axis=1)[:,np.newaxis],3,axis=1) )

    for i in range(npoints): 

        v0 = helpers.calculate_tangent_vector(patch_onsphere[i], patch_onsphere[(i-1)%npoints])
        v1 = helpers.calculate_tangent_vector(patch_onsphere[i], patch_onsphere[(i+1)%npoints])

        interior_angle_sum += np.arccos(np.dot(v0,v1))

    factor = interior_angle_sum - (len(patch_points)-2)*np.pi

    return factor / (np.pi*source_area)
