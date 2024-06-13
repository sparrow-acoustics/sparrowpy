import numpy as np
import numba

    
@numba.njit()
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
    pts_out = np.array([inner(rot_mat,p) for p in pts])

    return pts_out

@numba.njit()
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
def inner(matrix: np.ndarray,vector:np.ndarray)->np.ndarray:

    out = np.empty(matrix.shape[0])

    for i in numba.prange(matrix.shape[0]):
        out[i] = np.dot(matrix[i],vector)

    return out
