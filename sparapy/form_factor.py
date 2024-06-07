import numpy as np
from math import pi as PI
import sparapy.ff_helpers.geom as geom
import sparapy.ff_helpers.sampling as sampling
from sparapy.ff_helpers.integrate_line import poly_estimation,poly_integration
from sparapy.geometry import Polygon 
import numba as nb
from numba import njit
import matplotlib.pyplot as plt

#######################################################################################
### main
#######################################################################################
def calc_form_factor(receiving_patch: Polygon, emitting_patch:Polygon, mode='adaptive'):

    match mode:
        case 'adaptive':
            if singularity_check(receiving_patch.pts, emitting_patch.pts):
                return nusselt_integration(patch_i=emitting_patch.pts, patch_i_normal=emitting_patch.normal, patch_j=receiving_patch.pts, patch_j_normal=receiving_patch.normal, nsamples=64)
            else:
                return stokes_integration(patch_i=emitting_patch.pts, patch_j=receiving_patch.pts, source_area=emitting_patch.area,  approx_order=4)
        case 'naive':
            return naive_integration()


#######################################################################################
### form functions
#######################################################################################

def ffunction(p0,p1,n0,n1):
    cos0=geom.vec_cos(p1-p0, n0)
    cos1=geom.vec_cos(p0-p1,n1)
    rsq = np.square(np.linalg.norm(p0-p1))

    if rsq != 0:
        return(cos0*cos1/(rsq*PI))
    else:
        return 0
    
def stokes_ffunction(p0,p1):
    n = np.linalg.norm(p1-p0)

    if n>0:
        return np.log(n)
    else: # handle singularity (likely unused)
        return None

#######################################################################################
### integration
#######################################################################################

def naive_integration(patch_i: Polygon, patch_j: Polygon, n_samples=4, random=False):
    """
    calculate an estimation of the form factor between two patches 
    by computationally integrating the form function over the two patch surfaces

    The method is called naïve because it consists on the simplest approximation:
    the form function value is merely multiplied by the area of finite sub-elements of each patch's surface.

    Parameters
    ----------
    patch_i : geometry.Polygon
        radiance emitting patch

    patch_j : geometry.Polygon
        radiance receiving patch

    n_samples : int
        number of surface function samples on each patch 
        TO DO: convert to sample density factor (large and small patches have approx. same resolution)

    random: bool
        determines whether the form function is sampled at regular intervals (False) or randomly (uniform distribution)
        over the patches' surfaces.

    """

    if random:
        surfsampling = sampling.sample_random
    else:
        surfsampling = sampling.sample_regular

    int_accum = 0. # initialize integral accumulator variable

    i_samples = surfsampling(patch_i, npoints=n_samples)
    j_samples = surfsampling(patch_j, npoints=n_samples)

    for i_pt in i_samples:
        for j_pt in j_samples:
            int_accum+= ffunction(i_pt, j_pt, patch_i.normal, patch_j.normal)
  
    return int_accum * patch_j.area / ( len(j_samples) * len(j_samples) )

def naive_pt_integration(pt: np.ndarray, patch: Polygon, mode='source', n_samples=4, random=False):
    """


    """

    if random:
        surfsampling = sampling.sample_random
    else:
        surfsampling = sampling.sample_regular

    int_accum = 0. # initialize integral accumulator variable

    patch_samples = surfsampling(patch.pts, npoints=n_samples)

    if mode == 'source':
        source_area = patch.area
    elif mode == 'receiver':
        source_area = 1

    for patch_pt in patch_samples:
        int_accum+= ffunction(pt, patch_pt, patch_pt-pt, patch.normal)*(patch.area/len(patch_samples))
  
    return int_accum / source_area

def stokes_integration(patch_i: np.ndarray, patch_j: np.ndarray, source_area: float, approx_order=4):
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

    i_bpoints, i_conn = sampling.sample_border(patch_i, npoints=approx_order+1)
    j_bpoints, j_conn = sampling.sample_border(patch_j, npoints=approx_order+1)

    if singularity_check(i_bpoints,j_bpoints): 
        return float('nan')

    # first compute and store form function sample values
    form_mat = load_stokes_entries(i_bpoints, j_bpoints)


    # double polynomial integration (per dimension (x,y,z))
    outer_integral = 0
    inner_integral = np.zeros(shape=[len(i_bpoints),len(j_bpoints[0])])

    for dim in range(len(j_bpoints[0])):                                # for each dimension
        # integrate form function over each point on patch i boundary

        for i in range(len(i_bpoints)):                                 # for each point in patch i boundary
            for segj in j_conn:                                         # for each segment segj in patch j boundary
                
                xj = j_bpoints[segj][:,dim]          

                if xj[-1]-xj[0]!=0:
                    quadfactors = poly_estimation(xj,[form_mat[i][segj[k]] for k in range(len(segj))] ) # compute polynomial coefficients of approx form function over boundary segment xj
                    inner_integral[i][dim] += poly_integration(quadfactors,xj)                          # analytical integration of the approx polynomial


        # integrate previously computed integral over each boundary segment of patch i

        for segi in i_conn:                     # for each segment segi in patch i boundary

            xi = i_bpoints[segi][:,dim]

            if xi[-1]-xi[0]!=0:
                quadfactors = poly_estimation(xi,[inner_integral[segi[k]][dim] for k in range(len(segi))] ) 
                outer_integral += poly_integration(quadfactors,xi)


    return abs(outer_integral/(2*PI*source_area))


def nusselt_integration(patch_i: np.ndarray, patch_j: np.ndarray, patch_i_normal: np.ndarray, patch_j_normal: np.ndarray, nsamples=2, random=False, plotflag=False):
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
        p0_array = sampling.sample_random(patch_j,nsamples)
    else:
        p0_array = sampling.sample_regular(patch_j,nsamples)

    out = 0

    for p0 in p0_array:
        out += nusselt_analog(surf_origin=p0, surf_normal=patch_j_normal, patch_points=patch_i, patch_normal=patch_i_normal)

    return out / (PI * len(p0_array))

def nusselt_analog(surf_origin, surf_normal, patch_points, patch_normal, plotflag=False) -> float:

    boundary_points, connectivity = sampling.sample_border(patch_points, npoints=3) # 3 points per boundary segment (for quadratic approximation)

    hand = np.sign(np.inner(np.cross( patch_points[1]-patch_points[0] , patch_points[2]-patch_points[1] ), patch_normal) )

    curved_area = 0
        
    sphPts = np.empty_like( boundary_points )
    plnPts = np.empty( shape=(len(boundary_points),2) )

    for ii in range(len(boundary_points)):
        sphPts[ii] = ((boundary_points[ii]-surf_origin)/np.linalg.norm(boundary_points[ii]-surf_origin)) # patch j points projected on the hemisphere

    for ii in range(len(sphPts)):
        plnPts[ii,:] = np.inner(geom.rotation_matrix(surf_normal),sphPts[ii])[:-1] # points on the hemisphere projected onto 

    if plotflag:
        fig, ax = plt.subplots()
        circle = plt.Circle((0,0), 1, color='blue', alpha=0.1)
        ax.add_patch(circle)
        ax.plot(plnPts[:,0] , plnPts[:,1], 'r*')
        ax.grid()
        ax.set_aspect('equal')
        plt.xlim([-1.2,1.2])
        plt.ylim([-1.2,1.2])
        plt.show()

    polygon_area(plnPts[0::2])

    for segmt in connectivity:

        if abs(np.cross(plnPts[segmt[-1]],plnPts[segmt[0]])) > 1e-12:

            if np.inner( plnPts[segmt[-1]], plnPts[segmt[0]] ) >= 0:                    # if the points on the segment span less than 90 degrees relative to the origin
                curved_area += area_under_curve(plnPts[segmt],order=2,plotflag=plotflag)

            else:                                                                       # if points span over 90º, additional sampling is required
                mpoint = sphPts[segmt[0]] + (sphPts[segmt[-1]] - sphPts[segmt[0]]) / 2
                marc = mpoint/np.linalg.norm(mpoint) # midpoint on the arc projected on the hemisphere
                a = sphPts[segmt[0]] + (marc - sphPts[segmt[0]]) / 2
                b = marc + (sphPts[segmt[-1]] - marc) / 2

                mpoint = np.inner(geom.rotation_matrix(surf_normal),mpoint)[:-1]
                marc = np.inner(geom.rotation_matrix(surf_normal),marc)[:-1]
                a = a/np.linalg.norm(a)
                a = np.inner(geom.rotation_matrix(surf_normal),a)[:-1]

                b = b/np.linalg.norm(b)
                b = np.inner(geom.rotation_matrix(surf_normal),b)[:-1]

                if plotflag:
                    ax.plot(a[0] , a[1], 'g*')
                    ax.plot(b[0] , b[1], 'g*')
                    ax.plot(marc[0] , marc[1], 'b*')


                linArea = np.linalg.norm(plnPts[segmt[-1]] - plnPts[segmt[0]]) * np.linalg.norm(mpoint-marc)/2
                left =  area_under_curve(np.array([plnPts[segmt[0]],a,marc]), order=2, plotflag=plotflag)
                right = area_under_curve(np.array([marc,b,plnPts[segmt[-1]]]), order=2, plotflag=plotflag)
                curved_area += hand*(linArea * np.sign(left) + left + right)

    return polygon_area(plnPts[0::2]) + curved_area

#######################################################################################
### helpers
#######################################################################################

def load_stokes_entries(i_bpoints, j_bpoints):
    form_mat = [[[] for j in range(len(j_bpoints))] for i in range(len(i_bpoints))]

    for j,pj in enumerate(j_bpoints):
        for i,pi in enumerate(i_bpoints):
            form_mat[i][j] = stokes_ffunction(pi,pj)

    return form_mat


def singularity_check(p0,p1):
    """
    returns true if two patches have any common points
    """

    for point in p1:
        if (p0==point).all(axis=1).any():
            return True

    return False

def polygon_area(pts: np.ndarray):
    """
    calculates the area of a convex 3- or 4-sided polygon

    Parameters
    ----------
    pts: np.ndarray
        list of 3D points which define the vertices of the polygon
    """

    if len(pts) == 3:
        return .5 * np.linalg.norm( np.cross( pts[1]-pts[0] , pts[2]-pts[0] ) )

    if len(pts) == 4:
        return .5 * ( np.linalg.norm( np.cross( pts[3]-pts[2] , pts[0]-pts[2] ) ) + np.linalg.norm( np.cross( pts[1]-pts[0] , pts[2]-pts[0] ) ) )

def area_under_curve(ps, order=2, plotflag=False):
    """
    calculates the area under a polynomial curve sampled by a finite number of points (on a shared plane)
    
    Parameters
    ----------
    ps : np.ndarray
        sample points

    order : int
        polynomial order of the curve
    """

    order = min(order,len(ps)-1) # the order of the curve may be overwritten depending on the sample size

    f  = ps[-1] - ps[0] # the vector between first and last sample (y==0) (new space's x axis)

    rotation_matrix = np.array([[f[0],f[1]],[-f[1],f[0]]])/np.linalg.norm(f) 

    x = np.array([0]) # origin of new linear space
    y = np.array([0])

    for k in range(1,order+1):

        c = ps[k] - ps[0]                   # translate point towards new origin
        cc = np.inner(rotation_matrix,c)    # rotate point around origin to align with new axis
        
        x = np.append(x,cc[0])
        y = np.append(y,cc[1])


    coefs = poly_estimation(x,y)
    area = poly_integration(coefs,x)        # area between curve and ps[-1] - ps[0]

    if plotflag:
        ax = plt.gca()

        xx = np.linspace(x[0], x[-1], 100)
        yy = np.zeros_like(xx)

        for o in range(len(ps)):
            yy += coefs[o] * xx**( len(ps) - (o+1) )

        smoothpts = np.transpose(np.transpose(np.inner( np.transpose(rotation_matrix), np.transpose(np.array([xx,yy])) )) + ps[0])

        ax.plot(smoothpts[0], smoothpts[1], 'r--')



    return area