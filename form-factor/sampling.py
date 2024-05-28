import numpy as np
from elmt import elmt
from geometry import Polygon as polyg

def sample_random(el: polyg, npoints=100):
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
    
    u = el.pts[1]-el.pts[0]
    v = el.pts[-1]-el.pts[0]

    for i in range(npoints):
        s = np.random.uniform()
        t = np.random.uniform()

        inside = s+t <= 1

        if len(el.pts)==3 and not inside: # if sample falls outside of triangular patch, it is "reflected" inside 
            s = 1-s
            t = 1-t
        
        ptlist[i] = s*u + t*v + el.pts[0]

    return ptlist

def sample_regular(el: polyg, npoints=10):
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

    u = el.pts[1]-el.pts[0]
    v = el.pts[-1]-el.pts[0] 

    if len(el.pts)==3:
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

    tt = np.linspace(0,1,npointsx, endpoint=False)
    sstep =  1/(npointsx*2)
    tt += sstep

    tz = np.linspace(0,1,npointsz, endpoint=False)
    sstepz =  1/(npointsz*2)
    tz += sstepz

    thres = np.sqrt(sstepz**2+sstep**2)/2

    jj = 0

    # TO DO: find way to sample triangles more evenly 

    for i,s in enumerate(tt):
        if len(el.pts)==3:
            jj = i

        for t in tz[0:len(tz)-round(npointsz/npointsx*jj)]: 

            inside = s+t <= 1-thres
            if not(len(el.pts)==3 and not inside):
                ptlist.append(s*u + t*v + el.pts[0])

    return np.array(ptlist)

def sample_border(el: polyg,npoints=3):
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

    pts = np.empty((len(el.pts)*(npoints-1),len(el.pts[0])))
    conn=[[[] for i in range(npoints)] for j in range(len(el.pts))]

    for i in range(len(el.pts)):

        conn[i][0]= (i*n_div)%(n_div*len(el.pts))
        conn[i][-1]= (i*n_div+n_div)%(n_div*len(el.pts))

        for ii in range(0,n_div):

            pts[i*n_div+ii,:]= (el.pts[i] + ii*(el.pts[(i+1)%len(el.pts)]-el.pts[i])/n_div)

            conn[i][ii]=(i*n_div+ii)%(n_div*len(el.pts))


    return pts,conn