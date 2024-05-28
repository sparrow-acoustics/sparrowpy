import numpy as np
from integrate_line import poly_estimation, poly_integration
import sampling
import geom

from geometry import Polygon as polyg

PI = np.pi

def nusselt(patch_i: polyg, patch_j: polyg, nsamples=2, random=False):
    """
    Estimates the form factor by integrating the Nusselt analogue values (emitting patch) 
    over the surface of the receiving patch

    Parameters
    ----------
    patch_i: geometry.Polygon
            receiving patch

    patch_j: geometry.Polygon
            emitting patch

    nsamples: int
            number of surface samples for integration

    random: bool
            determines the distribution of the samples on patch_i surface 
            if True, the samples are randomly distributed in a uniform way
            if False, a regular sampling of the surface is performed

    sphRadius: float
            radius of the hemisphere used for the Nusselt analog estimation
            (likely irrelevant)

    """

    if random:
        p0_array = sampling.sample_random(patch_i,nsamples)
    else:
        p0_array = sampling.sample_regular(patch_i,nsamples)

    out = 0

    b_points, connectivity = sampling.sample_border(patch_j, npoints=3) # 3 points per boundary segment (for quadratic approximation)

    sphPts = np.empty_like( b_points )
    plnPts = np.empty( shape=(len(b_points),2) )

    for p0 in p0_array:

        curved_area = 0
 
        
        for ii in range(len(b_points)):
            sphPts[ii] = ((b_points[ii]-p0)/np.linalg.norm(b_points[ii]-p0)) # patch j points projected on the hemisphere

        
        for ii in range(len(sphPts)):
            plnPts[ii,:] = np.inner(geom.rotation_matrix(patch_i.normal),sphPts[ii])[:-1] # points on the hemisphere projected onto 

        projPolyArea = polygon_area(plnPts[0::2])
            
        for segmt in connectivity:

            if np.inner( plnPts[segmt[-1]], plnPts[segmt[0]] ) >= 0:                    # if the points on the segment span less than 90 degrees relative to the origin
                curved_area += area_under_curve(plnPts[segmt],order=2)

            else:                                                                       # if points span over 90º, additional sampling is required
                mpoint = sphPts[segmt[0]] + (sphPts[segmt[-1]] - sphPts[segmt[0]]) / 2
                marc = mpoint/np.linalg.norm(mpoint) # midpoint on the arc projected on the hemisphere

                mpoint = np.inner(geom.rotation_matrix(patch_i.normal),mpoint)[:-1]
                marc = np.inner(geom.rotation_matrix(patch_i.normal),marc)[:-1]

                linArea = np.linalg.norm(plnPts[segmt[-1]] - plnPts[segmt[0]]) * np.linalg.norm(mpoint-marc)/2
                
                a = sphPts[segmt[0]] + (sphPts[segmt[1]] - sphPts[segmt[0]]) / 2
                a = a/np.linalg.norm(a)
                a = np.inner(geom.rotation_matrix(patch_i.normal),a)[:-1]

                b = sphPts[segmt[1]] + (sphPts[segmt[-1]] - sphPts[segmt[1]]) / 2
                b = b/np.linalg.norm(b)
                b = np.inner(geom.rotation_matrix(patch_i.normal),b)[:-1]

                left =  area_under_curve(np.array([plnPts[segmt[0]],a,marc]),order=2)
                right = area_under_curve(np.array([marc,b,plnPts[segmt[-1]]]),order=2)
                curved_area += linArea * np.sign(left) + left + right

        out += (projPolyArea + curved_area) / PI * (patch_i.A/len(p0_array)) / patch_i.A
       
    return out

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