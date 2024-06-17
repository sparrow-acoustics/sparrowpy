import numpy as np
import numba
import matplotlib.pyplot as plt

###################################################
# integration
################# 1D , polynomial
@numba.njit()
def poly_estimation(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    xmat = np.empty((len(x),len(x)))

    if x[-1]-x[0]==0:
        b = np.zeros(len(x))
    else:
        for i,xi in enumerate(x):
            for o in range(len(x)):
                xmat[i,len(x)-1-o] = xi**o
        
        b = inner(np.linalg.inv(xmat), y)

    return b

@numba.njit()
def poly_integration(c: np.ndarray, x: np.ndarray)-> float:

    out = 0

    s = -np.sign(x[-1]-x[0])

    for j in [-1,0]:
        for i in range(len(c)):
            out -= s * -1**j * c[i] * x[j]**(len(c)-i) / (len(c)-i)

    return out

################# surface areas
@numba.njit()
def polygon_area(pts: np.ndarray) -> float:
    """
    calculates the area of a convex n-sided polygon

    Parameters
    ----------
    pts: np.ndarray
        list of 3D points which define the vertices of the polygon
    """

    area = 0

    for tri in range(pts.shape[0]-2):
        area  +=  .5 * np.linalg.norm(np.cross(pts[tri+1] - pts[0], pts[tri+2]-pts[0]))
    
    return area

@numba.njit()
def area_under_curve(ps: np.ndarray, order=2) -> float:
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

    x = np.zeros(order+1)
    y = np.zeros(order+1)

    for k in range(1,order+1):

        c = ps[k] - ps[0]                   # translate point towards new origin
        cc = inner(matrix=rotation_matrix,vector=c)    # rotate point around origin to align with new axis
        
        x[k] = cc[0]
        y[k] = cc[1]


    coefs = poly_estimation(x,y)
    area = poly_integration(coefs,x)        # area between curve and ps[-1] - ps[0]

    return area

####################################################
# sampling
################# surface
@numba.njit()
def sample_random(el: np.ndarray, npoints=100):
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
    
    u = el[1]-el[0]
    v = el[-1]-el[0]

    for i in range(npoints):
        s = np.random.uniform()
        t = np.random.uniform()

        inside = s+t <= 1

        if len(el)==3 and not inside: # if sample falls outside of triangular patch, it is "reflected" inside 
            s = 1-s
            t = 1-t
        
        ptlist[i] = s*u + t*v + el[0]

    return ptlist

@numba.njit()
def sample_regular(el: np.ndarray, npoints=10):
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

    u = el[1]-el[0]
    v = el[-1]-el[0] 

    if len(el)==3:
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

    tt = np.linspace(0,1-1/npointsx,npointsx)
    sstep =  1/(npointsx*2)
    tt += sstep

    tz = np.linspace(0,1-1/npointsz,npointsz)
    sstepz =  1/(npointsz*2)
    tz += sstepz

    thres = np.sqrt(sstepz**2+sstep**2)/2

    jj = 0

    # TO DO: find way to sample triangles more evenly 

    for i,s in enumerate(tt):
        if len(el)==3:
            jj = i

        for t in tz[0:len(tz)-round(npointsz/npointsx*jj)]: 

            inside = s+t <= 1-thres
            if not(len(el)==3 and not inside):
                ptlist.append(s*u + t*v + el[0])


    out = np.empty((len(ptlist), len(ptlist[0])))

    for i in numba.prange(len(ptlist)):
        for j in numba.prange(len(ptlist[0])):
            out[i][j] = ptlist[i][j]
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(out[:,0],out[:,1],out[:,2])
    # plt.show()


    return out

################# boundary
@numba.njit()
def sample_border(el: np.ndarray, npoints=3):
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
@numba.njit()
def inner(matrix: np.ndarray,vector:np.ndarray)->np.ndarray:

    out = np.empty(matrix.shape[0])

    for i in numba.prange(matrix.shape[0]):
        out[i] = np.dot(matrix[i],vector)

    return out

@numba.njit()
def rotation_matrix(n_in: np.ndarray, n_out=np.array([])):
    """
    Computes a rotation matrix from a given input vector and desired output direction

    TO DO: expand to N-D arrays

    Parameters
    ----------
    n_in : numpy.ndarray(3,) 
        input vector
            
    n_out : numpy.ndarray(3,)
        direction to which n_in is to be rotated
    """

    if n_out.shape[0] == 0:
        n_out = np.zeros_like(n_in)
        n_out[-1] = 1.
    else:
        n_out = n_out

    #check if all the vector entries coincide
    counter = int(0)

    for i in numba.prange(n_in.shape[0]):
        if n_in[i] == n_out[i]:
            counter+=1
        else:
            counter=counter
        

    if counter == n_in.shape[0]:               # if input vector is the same as output return identity matrix

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
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            matrix =  np.eye( len(n_in) ) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

        else: # in case the in and out vectors have symmetrical directions
            matrix = np.array([[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]])

    return matrix

@numba.njit()
def calculate_tangent_vector(v0: np.ndarray, v1:np.ndarray) -> np.ndarray:
    
    if np.dot(v0,v1)!=0:
        scale = np.sqrt( np.square( np.linalg.norm(np.cross(v0,v1))/np.dot(v0,v1) ) + np.square( np.linalg.norm(v0) ) )

        vout = v1*scale - v0
        vout /= np.linalg.norm(vout)

    else:
        vout = v1/np.linalg.norm(v1)

    return vout

####################################################
# checks
@numba.njit()
def coincidence_check(p0: np.ndarray, p1: np.ndarray) -> bool:
    """
    returns true if two patches have any common points
    """
    
    flag = False
    
    for i in numba.prange(p0.shape[0]):
        for j in numba.prange(p1.shape[0]):
            count=0
            for k in numba.prange(p0.shape[1]):
                if p0[i,k]==p1[j,k]:
                    count+=1
                else:
                    pass

            if count == p0.shape[1]:
                flag=True
            else:
                pass

    return flag