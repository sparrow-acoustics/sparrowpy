"""Functions for computation of visibility matrix/factors."""
import numpy as np
import numba
from sparapy.radiosity_fast.universal_ff.ffhelpers import rotation_matrix,inner
import matplotlib.pyplot as plt

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
            if point_in_polygon(point3d=pt, polygon3d=surf_points,
                                plane_normal=surf_normal):
                is_visible = 0

    return is_visible

#@numba.njit()
# def compile_visible_patch(vis_point: np.ndarray,
#                            patch_points: np.ndarray, patch_normal: np.ndarray,
#                            surf_points: np.ndarray):
#     """Find and list interaction points between patch and a given surface."""
#     vis_patch = np.empty((0,0,0))

#     transf_surf = np.empty_like(surf_points)
#     verts_in = np.array([],dtype=bool)

#     calculate = True

#     for i in numba.prange(surf_points.shape[0]):
#         transf_surf[i] = project_to_plane(origin=vis_point,
#                                           point=surf_points[i],
#                                           plane_pt=patch_points[0],
#                                           plane_normal=patch_normal)
#         if transf_surf is None:
#             calculate = False
#             break

#         verts_in = np.append(verts_in, point_in_polygon(transf_surf[i],
#                                                         patch_points,
#                                                         patch_normal))


#     intersections = find_all_intersections(poly1=patch_points,
#                                             poly2=transf_surf)

#@numba.njit()
def poly_union(poly1: np.ndarray, poly2: np.ndarray,
                                  normal:np.ndarray) -> np.ndarray:
    """Generate oriented point list describing the union of two polygons.

    The polygons must be coplanar

    Parameters
    ----------
    poly1: np.ndarray (N_vertices_1, 3)
        Vertex array of first polygon

    poly2: np.ndarray (N_vertices_2, 3)
        Vertex array of second polygon

    normal: np.ndarray (3,)
        normal of the polygons' plane

    Returns
    -------
    poly_out: np.ndarray (N_vertices_out, 3)
        Vertex array of resulting polygon

    """
    int_list = find_all_intersections(poly1=poly1, poly2=poly2)

    int_conn=int_list[:,:2].astype(int)
    int_points = int_list[:,2:]


    counter = 0

    while counter+2 < int_conn.shape[0]:
        if int_conn[counter,0]==int_conn[counter+2,0] and int_conn[counter,0]==int_conn[counter+3,0]:
            int_conn=np.reshape(np.append(int_conn.flatten()[:((counter+1)*2)],int_conn.flatten()[((counter+3)*2):]),(-1,2))
            int_points=np.reshape(np.append(int_points.flatten()[:((counter+1)*3)],int_points.flatten()[((counter+3)*3):]),(-1,3))
        elif int_conn[counter,1]==int_conn[counter+2,1] and int_conn[counter,1]==int_conn[counter+3,1]:
            int_conn=np.reshape(np.append(int_conn.flatten()[:((counter+1)*2)],int_conn.flatten()[((counter+3)*2):]),(-1,2))
            int_points=np.reshape(np.append(int_points.flatten()[:((counter+1)*3)],int_points.flatten()[((counter+3)*3):]),(-1,3))
        else:
            counter+=2

    poly_out = np.array([poly1[0]])

    for kk in range(int_conn.shape[0]):

        if kk%2==0:
            s=1
            pol=poly2
        else:
            s=0
            pol=poly1

        ii = int_conn[kk,s]

        jj = int_conn[(kk+1)%int_conn.shape[0],s]

        if jj <= ii and not (jj==0):
            search = np.arange(jj-1,ii)%pol.shape[0]
        elif jj <= ii and (jj==0):
            search = np.arange(ii+1,jj+1+pol.shape[0])%pol.shape[0]
        else:
            search = np.arange(ii+1,jj+1)%pol.shape[0]

        poly_out = np.concatenate((poly_out,int_points[kk:kk+1], pol[search]))

        # print(int_conn[kk])
        # print(".")
        # print(search)
        # print("...")

        # plt.figure()
        # polllly=poly_out[1:,:2].tolist()
        # xs, ys = zip(*polllly)
        # plt.xlim([-1.5,1.5])
        # plt.ylim([-1.5,1.5])
        # plt.grid()
        # plt.plot(xs,ys)
        # plt.show()

    return poly_out[1:]



@numba.njit()
def find_all_intersections(poly1:np.ndarray,
                            poly2:np.ndarray) -> np.ndarray:
    """Find all intersections between two coplanar polygons."""
    out = np.empty((1,5))

    for i in range(poly1.shape[0]):
        p0 = poly1[i]
        p1 = poly1[(i+1)%poly1.shape[0]]

        dist = np.empty((0))
        temp = np.empty((0))

        for j in range(poly2.shape[0]):
            s0 = poly2[j]
            s1 = poly2[(j+1)%poly2.shape[0]]
            int_pt = line_line_int(p0,p1,s0,s1)

            if int_pt.shape[0]==3:
                temp = np.append(temp, np.array([float(i),float(j)]))
                temp = np.append(temp, int_pt)
                dist = np.append(dist, np.linalg.norm(int_pt-p0))

        temp = temp.reshape((-1,5))
        out = np.concatenate((out,temp[np.argsort(dist)]))

    return out[1:]


@numba.njit()
def line_line_int(a,b,c,d):
    """Calculate point of intersection between two lines in 2D."""
    out = np.empty((0,))
    k=np.empty((2,a.shape[0]))
    k[0] = b-a
    k[1] = c-d

    if np.all((d-a)==0.):
        temp=c
        c=d
        d=temp

    norm = np.array([np.linalg.norm(k[0]),np.linalg.norm(k[1])])
    normal = np.cross(b-a,d-a)/np.linalg.norm(np.cross(b-a,d-a))

    rot_mat = rotation_matrix(n_in=normal)

    k[0] = inner(matrix=rot_mat, vector=k[0])
    k[1] = inner(matrix=rot_mat, vector=k[1])
    bb   = inner(matrix=rot_mat, vector=c-a)[:2]

    k = k[:,:2]

    if abs(np.dot(k[0]/norm[0],k[1]/norm[1]))<1-1e-6:
        t = np.linalg.solve(k.T, bb)
        p = (1-t[0])*a + t[0]*b
        if (np.linalg.norm(p-a)<=np.linalg.norm(b-a) and
            np.linalg.norm(p-b)<=np.linalg.norm(b-a) and
            np.linalg.norm(p-c)<=np.linalg.norm(d-c) and
            np.linalg.norm(p-d)<=np.linalg.norm(d-c)):
            out = p

    return out

@numba.njit()
def project_to_plane(origin: np.ndarray, point: np.ndarray,
                     plane_pt: np.ndarray, plane_normal: np.ndarray,
                     epsilon=1e-6, check_normal=True):
    """Project point onto plane following direction defined by origin."""
    v = point-origin
    dotprod = np.dot(v,plane_normal)

    cond = dotprod < -epsilon

    if not check_normal:
        cond = abs(dotprod) > epsilon

    if cond:
        w = point-plane_pt
        fac = -np.dot(plane_normal,w) / dotprod

        int_point = w + plane_pt + fac*v
    else:
        int_point = None

    return int_point

@numba.njit()
def point_in_polygon(point3d: np.ndarray,
                     polygon3d: np.ndarray, plane_normal: np.ndarray,
                     eta=1e-6) -> bool:
    """Check if point is inside given polygon."""
    rotmat = rotation_matrix(n_in=plane_normal)

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


