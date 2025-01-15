"""Functions for computation of visibility matrix/factors."""
import numpy as np
import numba
from sparrowpy.radiosity_fast.universal_ff.ffhelpers import rotation_matrix,inner

@numba.njit()
def basic_visibility(vis_point: np.ndarray,
                     eval_point: np.ndarray,
                     surf_points: np.ndarray, surf_normal: np.ndarray,
                     eta=1e-6)->bool:
    """Return visibility of a point based on view point position.

    Parameters
    ----------
    vis_point: np.ndarray (3,)
        view point from which the visibility is being evaluated.

    eval_point: np.ndarray (3,)
        point being evaluated for visibility.

    surf_points: np.ndarray (N,3)
        boundary points of a possibly blocking surface.

    surf_normal: np.ndarray (3,)
        normal of possibly blocking surface.

    eta: float
        coplanarity check tolerance

    Returns
    -------
    is_visible: bool
        point visibility flag.
        (1 if points are visible to eachother, otherwise 0)

    """
    is_visible = True

    # if eval point is not coplanar with surf
    if np.abs(np.dot(surf_normal,eval_point-surf_points[0]))>eta:

        # check if projected point on surf
        pt = project_to_plane(origin=vis_point, point=eval_point,
                            plane_pt=surf_points[0],
                            plane_normal=surf_normal,
                            check_normal=True)

        # if intersection point exists
        if pt is not None:
            # if plane is in front of eval point
            if (np.linalg.norm(eval_point-vis_point)>
                                np.linalg.norm(pt-vis_point)):
                # if point is inside surf polygon
                if point_in_polygon(point3d=pt, polygon3d=surf_points,
                                    plane_normal=surf_normal):
                    is_visible = False

    # if both vis and eval point are coplanar
    elif (np.abs(np.dot(surf_normal,vis_point-surf_points[0]))<eta and
                    np.abs(np.dot(surf_normal,eval_point-surf_points[0]))<eta):
        is_visible = False




    return is_visible

@numba.njit()
def project_to_plane(origin: np.ndarray, point: np.ndarray,
                     plane_pt: np.ndarray, plane_normal: np.ndarray,
                     epsilon=1e-6, check_normal=True):
    """Project point onto plane following direction defined by origin.

    Also applicable to lower or higher-dimensional spaces, not just 3D.

    Parameters
    ----------
    origin: np.ndarray(float) (n_dims,)
        point determining projection direction

    point: np.ndarray(float) (n_dims,)
        point to project

    plane_pt: np.ndarray(float) (n_dims,)
        reference point on projection plane

    plane_normal: np.ndarray(float) (n_dims,)
        normal of projection plane

    epsilon: float
        tolerance for plane orthogonality check

    check_normal: bool
        if True, discards projections on planes facing
        opposite direction from the point and origin.

    Returns
    -------
    int_point: np.ndarray(float) (n_dims,) or None
        intersection point.
        None if no intersection point is found

    """
    v = point-origin
    dotprod = np.dot(v,plane_normal)

    cond = dotprod < -epsilon

    if not check_normal:
        cond = abs(dotprod) > epsilon

    if cond:
        w = point-plane_pt
        fac = -np.divide(np.dot(plane_normal,w), dotprod)

        int_point = w + plane_pt + fac*v
    else:
        int_point = None

    return int_point

@numba.njit()
def point_in_polygon(point3d: np.ndarray,
                     polygon3d: np.ndarray, plane_normal: np.ndarray,
                     eta=1e-6) -> bool:
    """Check if point is inside given polygon.

    Parameters
    ----------
    point3d: np.ndarray(float) (3,)
        point being evaluated.

    polygon3d: np.ndarray (N,3)
        polygon boundary points

    plane_normal: np.ndarray(float) (3,)
        normal of the polygon's plane

    eta: float
        tolerance for point inside of line check.

    Returns
    -------
    out: bool
        flags if point is inside polygon
        (True if inside, False if not)

    """
    # rotate all (coplanar) points to a horizontal plane
    # and remove z dimension for convenience
    rotmat = rotation_matrix(n_in=plane_normal)

    pt = inner(matrix=rotmat,vector=point3d)[0:point3d.shape[0]-1]
    poly = np.empty((polygon3d.shape[0],2))
    for i in numba.prange(polygon3d.shape[0]):
        poly[i] = inner(matrix=rotmat,vector=polygon3d[i])[0:point3d.shape[0]-1]


    # winding number algorithm
    count = 0
    for i in numba.prange(poly.shape[0]):
        a1 = poly[(i+1)%poly.shape[0]]
        a0 = poly[i%poly.shape[0]]
        side = a1-a0

        # check if line from evaluation point in +x direction
        # intersects polygon side
        nl = np.array([-side[1],side[0]])/np.linalg.norm(side)
        b = project_to_plane(origin=pt, point=pt+np.array([1.,0.]),
                             plane_pt=a1, plane_normal=nl,
                             check_normal=False)

        # check if intersection exists and if is inside side [a0,a1]
        if (b is not None) and b[0]>pt[0]:
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


