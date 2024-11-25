"""Functions for computation of visibility matrix/factors."""
import numpy as np
import numba
from sparapy.radiosity_fast.universal_ff.ffhelpers import rotation_matrix,inner

@numba.njit()
def basic_visibility(vis_point: np.ndarray,
                     patch_center: np.ndarray,
                     surf_points: np.ndarray, surf_normal: np.ndarray):
    """Return visibility 1/0 based on patch centroid position."""
    is_visible = 1

    pt = project_to_plane(origin=vis_point, point=patch_center,
                            plane_pt=surf_points[0], plane_normal=surf_normal)

    if pt is not None:
        if np.linalg.norm(patch_center-vis_point)>np.linalg.norm(pt-vis_point):
            if point_in_polygon(point3d=pt,
                                              polygon3d=surf_points,
                                              plane_normal=surf_normal):
                is_visible = 0

    return is_visible



@numba.njit()
def project_to_plane(origin: np.ndarray, point: np.ndarray,
                     plane_pt: np.ndarray, plane_normal: np.ndarray,
                     epsilon=1e-6, check_normal=True):
    """Project point onto plane following direction defined by origin."""
    v = point-origin
    dot = np.dot(v,plane_normal)

    cond = dot < -epsilon

    if not check_normal:
        cond = abs(dot) > epsilon

    if cond:
        w = point-plane_pt
        fac = -np.dot(plane_normal,w) / dot

        int_point = w + plane_pt + fac*v
    else:
        int_point = None

    return int_point

@numba.njit()
def point_in_polygon(point3d: np.ndarray,
                     polygon3d: np.ndarray, plane_normal: np.ndarray,
                     eta=1e-6):
    """Check if point is inside given polygon."""
    rotmat = rotation_matrix(n_in=plane_normal,n_out=np.array([0.,0.,1.]))

    pt = inner(matrix=rotmat,vector=point3d)[0:point3d.shape[0]-1]

    poly = np.empty((polygon3d.shape[0],2))
    for i in numba.prange(polygon3d.shape[0]):
        poly[i] = inner(matrix=rotmat,vector=polygon3d[i])[0:point3d.shape[0]-1]

    count = 0
    for i in numba.prange(poly.shape[0]):
        a1 = poly[(i+1)%poly.shape[0]]
        a0 = poly[i%poly.shape[0]]

        side = a1-a0
        nl = np.array([-side[1],side[0]])/np.linalg.norm(side)
        b = project_to_plane(origin=pt, point=pt+np.array([1.,0.]),
                             plane_pt=a1, plane_normal=nl,
                             check_normal=False)

        if (b is not None) and b[0]>pt[0]: # an intersection point is found
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


