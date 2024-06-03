import numpy as np
from math import pi as PI
import sparapy.ff_helpers.geom as geom
import sparapy.ff_helpers.sampling as sampling
from sparapy.ff_helpers.integrate_line import poly_estimation,poly_integration
from sparapy.geometry import Polygon 

#######################################################################################
### main
#######################################################################################
def calculate_form_factor(receiving_patch: Polygon, emitting_patch:Polygon, mode='adaptive'):

    match mode:
        case 'adaptive':
            if singularity_check(receiving_patch.pts, emitting_patch.pts):
                return nusselt_integration(patch_i=emitting_patch, patch_j=receiving_patch, nsamples=25)
            else:
                return stokes_integration(patch_i=emitting_patch, patch_j=receiving_patch, approx_order=2)


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
            int_accum+= ffunction(i_pt, j_pt, patch_i.normal, patch_j.normal)*(patch_i.area/len(i_samples))*(patch_j.area/len(j_samples))
  
    return int_accum/patch_i.area

def stokes_integration(patch_i: Polygon, patch_j: Polygon, approx_order=4):
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
    form_mat = [[[] for j in range(len(j_bpoints))] for i in range(len(i_bpoints))]

    for j,pj in enumerate(j_bpoints):
        for i,pi in enumerate(i_bpoints):
            form_mat[i][j] = stokes_ffunction(pi,pj)


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


    return abs(outer_integral/(2*PI*patch_i.area))

def nusselt_integration(patch_i: Polygon, patch_j: Polygon, nsamples=2, random=False):
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

    b_points, connectivity = sampling.sample_border(patch_i, npoints=3) # 3 points per boundary segment (for quadratic approximation)

    sphPts = np.empty_like( b_points )
    plnPts = np.empty( shape=(len(b_points),2) )

    for p0 in p0_array:

        curved_area = 0
 
        
        for ii in range(len(b_points)):
            sphPts[ii] = ((b_points[ii]-p0)/np.linalg.norm(b_points[ii]-p0)) # patch j points projected on the hemisphere

        
        for ii in range(len(sphPts)):
            plnPts[ii,:] = np.inner(geom.rotation_matrix(patch_j.normal),sphPts[ii])[:-1] # points on the hemisphere projected onto 

        projPolyArea = polygon_area(plnPts[0::2])
            
        for segmt in connectivity:

            if np.inner( plnPts[segmt[-1]], plnPts[segmt[0]] ) >= 0:                    # if the points on the segment span less than 90 degrees relative to the origin
                curved_area += area_under_curve(plnPts[segmt],order=2)

            else:                                                                       # if points span over 90º, additional sampling is required
                mpoint = sphPts[segmt[0]] + (sphPts[segmt[-1]] - sphPts[segmt[0]]) / 2
                marc = mpoint/np.linalg.norm(mpoint) # midpoint on the arc projected on the hemisphere

                mpoint = np.inner(geom.rotation_matrix(patch_j.normal),mpoint)[:-1]
                marc = np.inner(geom.rotation_matrix(patch_j.normal),marc)[:-1]

                linArea = np.linalg.norm(plnPts[segmt[-1]] - plnPts[segmt[0]]) * np.linalg.norm(mpoint-marc)/2
                
                a = sphPts[segmt[0]] + (sphPts[segmt[1]] - sphPts[segmt[0]]) / 2
                a = a/np.linalg.norm(a)
                a = np.inner(geom.rotation_matrix(patch_j.normal),a)[:-1]

                b = sphPts[segmt[1]] + (sphPts[segmt[-1]] - sphPts[segmt[1]]) / 2
                b = b/np.linalg.norm(b)
                b = np.inner(geom.rotation_matrix(patch_j.normal),b)[:-1]

                left =  area_under_curve(np.array([plnPts[segmt[0]],a,marc]),order=2)
                right = area_under_curve(np.array([marc,b,plnPts[segmt[-1]]]),order=2)
                curved_area += linArea * np.sign(left) + left + right

        out += (projPolyArea + curved_area) 
       
    return out * (patch_j.area/len(p0_array)) / (PI * patch_i.area)

#######################################################################################
### helpers
#######################################################################################

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

def area_under_curve(ps, order=2):
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

    return area