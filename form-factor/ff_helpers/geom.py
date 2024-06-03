import numpy as np



def vec_cos(v0,v1):
    """
    cosine of the angle between two arbitrary n-dimensional vectors
    returns 1 if any vector has length 0

    Parameters
    ----------
    v0, v1 : numpy.ndarray 
            vectors of equal (N,1) dimensions
    """
    if np.linalg.norm(v1)==0 or np.linalg.norm(v0)==0:
        return 1
    else:
        return np.inner(v0,v1)/(np.linalg.norm(v0)*np.linalg.norm(v1))
    

def universal_transform(o, u, pts):
    """
    Universal linear transformation. 
    Translates and rotates points to a new cartesian coordinate system

    Parameters
    ----------
    o : numpy.ndarray(3,)
        origin of the output coordinate system in the global coordinate system

    u : numpy.ndarray(3,)
        up vector of the output coordinate system (is rotated toward (0,0,1))
            
    pts : numpy.ndarray(N,3)
        list of  N points to be transformed to new coordinate system
    """


    # translate points towards origin
    pts = translation(o, pt_list=pts) 

    # compute rotation matrix
    rot_mat = rotation_matrix(n_in=u)

    # compute points rotated around origin
    pts_out = np.array([np.inner(rot_mat,p) for p in pts])

    return pts_out


def translation(origin, pt_list):
    """
    Translates points towards an origin (N-dimensional)

    Parameters
    ----------
    origin : numpy.ndarray(N,) or list
        origin point
            
    pt_list : numpy.ndarray(M,N) or list
        list of M points to be translated
    """
    return np.array(pt_list) - np.array([origin for i in range(len(pt_list))])


def rotation_matrix(n_in: np.ndarray, n_out=None):
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

    if n_out is None:
        n_out = np.zeros_like(n_in)
        n_out[-1] = 1.

    if (n_in == n_out).all():               # if input vector is the same as output return identity matrix
        return np.eye( len(n_in) )

    a, b = (n_in / np.linalg.norm(n_in)).reshape( len(n_in) ), (n_out / np.linalg.norm(n_out)).reshape( len(n_in) )

    c = np.dot(a,b)
    
    if c!=-1:
        v = np.cross(a,b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        matrix =  np.eye( len(n_in) ) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else: # in case the in and out vectors have symmetrical directions
        matrix = np.array([[-1,0,0],[0,1,0],[0,0,-1]])

    return matrix

######################################################################################################
def vec_plane_intersection(p0=np.array([0,0,0]), v0=np.array([0,0,1]), n=np.array([]), pn=np.array([])):

    vdot = np.dot(v0, n)

    if vdot!=0:
        dif = pn-p0
        t = np.dot(n,dif)/vdot
        return np.array(p0+v0*t)
    
    return None


def point_in_polygon(pt,el):

    eta = 10**-7

    if pt is None:
        return False

    #first reduce the problem to a 2D problem
    p0 = universal_transform( el.o, el.n, np.array(pt) )[0,:-1]
    el0 = universal_transform(el.o, el.n, el.pt)[:,:-1]


    count = 0
    for i in range(len(el0)):
        l = el0[(i+1)%len(el0)]-el0[i%len(el0)]
        nl = [-l[1],l[0]]/np.linalg.norm(l)

        b = vec_plane_intersection(p0=p0, v0=np.array([1,0]), n = nl, pn=el0[i%len(el0)]) # check if a line from p0 intersects with given side's "plane" 

        if (b is not None) and b[0]>p0[0]: # an intersection point is found
            if abs(np.linalg.norm(b-el0[i%len(el0)])+np.linalg.norm(b-el0[(i+1)%len(el0)]) - np.linalg.norm(el0[i%len(el0)]-el0[(i+1)%len(el0)])) <= eta: # if is within border segment length
                if np.dot(b-p0,nl)>0:
                    count+=1
                elif np.dot(b-p0,nl)<0:
                    count-=1


    if count != 0:
        return True
    else:        
        return False