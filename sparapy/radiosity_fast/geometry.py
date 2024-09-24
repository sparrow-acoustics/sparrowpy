"""Geometry functions for the radiosity fast solver."""
import numba
import numpy as np


@numba.njit()
def get_scattering_data_receiver_index(
        pos_i:np.ndarray, pos_j:np.ndarray,
        receivers:np.ndarray, wall_id_i:np.ndarray,
        ):
    """Get scattering data depending on previous, current and next position.

    Parameters
    ----------
    pos_i : np.ndarray
        current position of shape (3)
    pos_j : np.ndarray
        next position of shape (3)
    receivers : np.ndarray
        receiver directions of all walls of shape (n_walls, n_receivers, 3)
    wall_id_i : np.ndarray
        current wall id to get write directional data

    Returns
    -------
    scattering_factor: float
        scattering factor from directivity

    """
    n_patches = pos_i.shape[0] if pos_i.ndim > 1 else 1
    receiver_idx = np.empty((n_patches), dtype=np.int64)
    # scattering_idx = np.empty((n_patches), dtype=np.int64)
    for i in range(n_patches):
        diff_receiver = pos_i[i]-pos_j
        diff_receiver /= np.linalg.norm(
            diff_receiver)
        receiver_idx[i] = np.argmin(np.sum(
            (receivers[wall_id_i[i], :]-diff_receiver)**2, axis=-1),
            axis=-1)
        # scattering_idx[i] = scattering_index[wall_id_i[i]]

    return receiver_idx

@numba.njit()
def get_relative_angles(point:np.ndarray, origin:np.ndarray, normal:np.ndarray, up:np.ndarray):
    """Get scattering data depending on previous, current position.

    Parameters
    ----------
    point : np.ndarray
        cartesian point in space (3)
    origin : np.ndarray
        referential origin (global cartesian coordinates) (3)
    normal : np.ndarray
        referential normal vector (3)
    up : np.ndarray
        referential up vector(3)
    

    Returns
    -------
    angles: np.ndarray
        azimuth and elevation angles of point relative to referential

    """
    pt = point-origin / np.linalg.norm(point-origin)
    
    a = np.dot(normal,np.cross(up,pt))
    
    if a==0:
        azimuth = 0
    else:
        proj_pt = pt - np.dot(pt,normal)/np.dot(normal,normal)*normal
        proj_pt/=np.linalg.norm(proj_pt)
        
        azimuth = np.sign(a)*np.arccos(np.dot(proj_pt,up)) 
            
        if np.sign(a) < 0:
            azimuth += 2*np.pi
            
    elevation = np.arcsin(np.dot(pt,normal))
        
    return np.array([azimuth,elevation])

@numba.njit()
def get_scattering_data(
        pos_h:np.ndarray, pos_i:np.ndarray, pos_j:np.ndarray,
        sources:np.ndarray, receivers:np.ndarray, wall_id_i:np.ndarray,
        scattering:np.ndarray, scattering_index:np.ndarray, mode="nneighbor"):
    """Get scattering data depending on previous, current and next position.

    Parameters
    ----------
    pos_h : np.ndarray
        previous position of shape (3)
    pos_i : np.ndarray
        current position of shape (3)
    pos_j : np.ndarray
        next position of shape (3)
    sources : np.ndarray
        source directions of all walls of shape (n_walls, n_sources, 3)
    receivers : np.ndarray
        receiver directions of all walls of shape (n_walls, n_receivers, 3)
    wall_id_i : np.ndarray
        current wall id to get write directional data
    scattering : np.ndarray
        scattering data of shape (n_scattering, n_sources, n_receivers, n_bins)
    scattering_index : np.ndarray
        index of the scattering data of shape (n_walls)

    Returns
    -------
    scattering_factor: float
        scattering factor from directivity

    """
    diff_source = pos_h-pos_i
    diff_receiver = pos_i-pos_j
    # wall_id_i = int(patch_to_wall_ids[i])
    diff_source /= np.linalg.norm(diff_source)
    diff_receiver /= np.linalg.norm(diff_receiver)
    source_idx = np.argmin(np.sum(
        (sources[wall_id_i, :, :]-diff_source)**2, axis=-1))
    receiver_idx = np.argmin(np.sum(
        (receivers[wall_id_i, :]-diff_receiver)**2, axis=-1))
    out = scattering[scattering_index[wall_id_i],
        source_idx, receiver_idx, :]
    
    return out

#@numba.njit()
def get_scattering_data_dist(
        pos_h:np.ndarray, pos_i:np.ndarray, pos_j:np.ndarray, i_normal:np.ndarray, i_up: np.ndarray,
        sources:np.ndarray, receivers:np.ndarray, wall_id_i:np.ndarray,
        scattering:np.ndarray, scattering_index:np.ndarray, mode="nneighbor", threshold=0.0001, order=1):
    """Get scattering data depending on previous, current and next position.

    Parameters
    ----------
    pos_h : np.ndarray
        previous position of shape (3)
    pos_i : np.ndarray
        current position of shape (3)
    pos_j : np.ndarray
        next position of shape (3)
    sources : np.ndarray
        source angular directions of all walls of shape (n_walls, n_sources, 2)
    receivers : np.ndarray
        receiver angular directions of all walls of shape (n_walls, n_receivers, 2)
    wall_id_i : np.ndarray
        current wall id to get write directional data
    scattering : np.ndarray
        scattering data of shape (n_scattering, n_sources, n_receivers, n_bins)
    scattering_index : np.ndarray
        index of the scattering data of shape (n_walls)

    Returns
    -------
    scattering_factor: float
        scattering factor from directivity

    """
    h=get_relative_angles(point=pos_h, origin=pos_i, normal=i_normal, up=i_up)
    j=get_relative_angles(point=pos_j, origin=pos_i, normal=i_normal, up=i_up)   
    

    s_d = np.abs(sources-h)
    r_d = np.abs(receivers-j)

    s_d[s_d[:,0]>np.pi,0]=s_d[s_d[:,0]>np.pi,0]-np.pi
    r_d[r_d[:,0]>np.pi,0]=r_d[r_d[:,0]>np.pi,0]-np.pi

    s_dist = np.empty(s_d.shape[0])
    r_dist = np.empty(r_d.shape[0])
    
    for i in numba.prange(s_d.shape[0]):
        s_dist[i] = np.linalg.norm(s_d[i,:])
    for i in numba.prange(r_d.shape[0]):
        r_dist[i] = np.linalg.norm(r_d[i,:])

    if mode == "nneighbor": 
        source_idx = np.argmin(s_dist)
        receiver_idx = np.argmin(r_dist)
        out = scattering[scattering_index[wall_id_i]][
            source_idx, receiver_idx, :]
        
    elif mode == "inv_dist":

        if (s_dist < threshold).any():
            source_idx = np.array([np.argmin(s_dist)])
            w_s = np.array([1.])
        else:
            source_idx = np.argpartition(-s_dist,-3)[-3:]
            w_s = 1/(s_dist[source_idx]**order)

        if (r_dist < threshold).any():
            receiver_idx = np.array([np.argmin(r_dist)])
            w_r = np.array([1.])
        else:
            receiver_idx = np.argpartition(-r_dist,-3)[-3:]
            w_r = 1/(r_dist[receiver_idx]**order)
        
            
        out = np.zeros((scattering[scattering_index[wall_id_i]].shape[-1],))
        den = 0

        for i in numba.prange(source_idx.shape[0]):
            for j in numba.prange(receiver_idx.shape[0]):
                out += w_s[i]*w_r[j] * scattering[scattering_index[wall_id_i]][source_idx[i],receiver_idx[j]][:]
                den += w_s[i]*w_r[j]

        out /= den

    else:
        out=np.array([0.])
        
    return out


@numba.njit()
def get_scattering_data_source(
        pos_h:np.ndarray, pos_i:np.ndarray,
        sources:np.ndarray, wall_id_i:np.ndarray,
        scattering:np.ndarray, scattering_index:np.ndarray):
    """Get scattering data depending on previous, current position.

    Parameters
    ----------
    pos_h : np.ndarray
        previous position of shape (3)
    pos_i : np.ndarray
        current position of shape (3)
    sources : np.ndarray
        source directions of all walls of shape (n_walls, n_sources, 3)
    wall_id_i : np.ndarray
        current wall id to get write directional data
    scattering : np.ndarray
        scattering data of shape (n_scattering, n_sources, n_receivers, n_bins)
    scattering_index : np.ndarray
        index of the scattering data of shape (n_walls)

    Returns
    -------
    scattering_factor: float
        scattering factor from directivity

    """
    diff_source = pos_h-pos_i
    diff_source /= np.linalg.norm(diff_source)
    source_idx = np.argmin(np.sum(
        (sources[wall_id_i, :, :]-diff_source)**2, axis=-1))
    return scattering[scattering_index[wall_id_i], source_idx]
    

@numba.njit(parallel=True)
def check_visibility(
        patches_center:np.ndarray, patches_normal:np.ndarray) -> np.ndarray:
    """Check the visibility between patches.

    Parameters
    ----------
    patches_center : np.ndarray
        center points of all patches of shape (n_patches, 3)
    patches_normal : np.ndarray
        normal vectors of all patches of shape (n_patches, 3)

    Returns
    -------
    visibility_matrix : np.ndarray
        boolean matrix of shape (n_patches, n_patches) with True if patches
        can see each other, otherwise false

    """
    n_patches = patches_center.shape[0]
    visibility_matrix = np.empty((n_patches, n_patches), dtype=np.bool_)
    visibility_matrix.fill(False)
    indexes = []
    for i_source in range(n_patches):
        for i_receiver in range(n_patches):
            if i_source < i_receiver:
                indexes.append((i_source, i_receiver))
    indexes = np.array(indexes)
    for i in numba.prange(indexes.shape[0]):
        i_source = indexes[i, 0]
        i_receiver = indexes[i, 1]
        patches_parallel = np.abs(np.dot(
            patches_normal[i_source], patches_normal[i_receiver]) -1) < 1e-5
        same_dim = np.sum(
            patches_normal[i_source] * patches_center[i_source]) == np.sum(
                patches_normal[i_receiver] * patches_center[i_receiver])
        if i_source == i_receiver:
            visibility_matrix[i_source, i_receiver] = False
        elif patches_parallel and same_dim:
            visibility_matrix[i_source, i_receiver] = False
        else:
            visibility_matrix[i_source, i_receiver] = True
    return visibility_matrix

@numba.njit()
def _create_patches(polygon_points:np.ndarray, max_size):
    """Create patches from a polygon."""
    size = np.empty(polygon_points.shape[1])
    for i in range(polygon_points.shape[1]):
        size[i] = polygon_points[:, i].max() - polygon_points[:, i].min()
    patch_nums = np.array([int(n) for n in size/max_size])
    real_size = size/patch_nums
    if patch_nums[2] == 0:
        x_idx = 0
        y_idx = 1
    if patch_nums[1] == 0:
        x_idx = 0
        y_idx = 2
    if patch_nums[0] == 0:
        x_idx = 1
        y_idx = 2

    x_min = np.min(polygon_points.T[x_idx])
    y_min = np.min(polygon_points.T[y_idx])

    n_patches = patch_nums[x_idx]*patch_nums[y_idx]
    patches_points = np.empty((n_patches, 4, 3))
    i = 0
    for i_x in range(patch_nums[x_idx]):
        for i_y in range(patch_nums[y_idx]):
            points = polygon_points.copy()
            points[0, x_idx] = x_min + i_x * real_size[x_idx]
            points[0, y_idx] = y_min + i_y * real_size[y_idx]
            points[1, x_idx] = x_min + (i_x+1) * real_size[x_idx]
            points[1, y_idx] = y_min + i_y * real_size[y_idx]
            points[3, x_idx] = x_min + i_x * real_size[x_idx]
            points[3, y_idx] = y_min + (i_y+1) * real_size[y_idx]
            points[2, x_idx] = x_min + (i_x+1) * real_size[x_idx]
            points[2, y_idx] = y_min + (i_y+1) * real_size[y_idx]
            patches_points[i] = points
            i += 1

    return patches_points


@numba.njit()
def _calculate_center(points):
    return np.sum(points, axis=-2) / points.shape[-2]

@numba.njit()
def _calculate_size(points):
    vec1 = points[..., 0, :]-points[..., 1, :]
    vec2 = points[..., 1, :]-points[..., 2, :]
    return np.abs(vec1-vec2)

@numba.njit()
def _calculate_area(points):
    vec1 = points[..., 0, :]-points[..., 1, :]
    vec2 = points[..., 1, :]-points[..., 2, :]
    size = vec1-vec2
    return np.abs(
        size[..., 0]*size[..., 1] + size[..., 1]*size[..., 2] \
            + size[..., 0]*size[..., 2])


@numba.njit()
def process_patches(
        polygon_points_array: np.ndarray,
        walls_normal: np.ndarray,
        patch_size:float, n_walls:int):
    """Process the patches.

    Parameters
    ----------
    polygon_points_array : np.ndarray
        points of the polygon of shape (n_walls, 4, 3)
    walls_normal : np.ndarray
        wall normal of shape (n_walls, 3)
    patch_size : float
        maximal patch size in meters of shape (n_walls, 3).
    n_walls : int
        number of walls

    Returns
    -------
    patches_points : np.ndarray
        points of all patches of shape (n_patches, 4, 3)
    patches_normal : np.ndarray
        normal of all patches of shape (n_patches, 3)
    n_patches : int
        number of patches

    """
    n_patches = 0
    n_walls = polygon_points_array.shape[0]
    for i in range(n_walls):
        n_patches += total_number_of_patches(
            polygon_points_array[i, :, :], patch_size)
    patches_points = np.empty((n_patches, 4, 3))
    patch_to_wall_ids = np.empty((n_patches), dtype=np.int64)
    patches_per_wall = np.empty((n_walls), dtype=np.int64)

    for i in range(n_walls):
        polygon_points = polygon_points_array[i, :]
        patches_points_wall = _create_patches(
            polygon_points, patch_size)
        patches_per_wall[i] = patches_points_wall.shape[0]
        j_start = (np.sum(patches_per_wall[:i])) if i > 0 else 0
        j_end = np.sum(patches_per_wall[:i+1])
        patch_to_wall_ids[j_start:j_end] = i
        patches_points[j_start:j_end, :, :] = patches_points_wall
    n_patches = patches_points.shape[0]

    # calculate patch information
    patches_normal = walls_normal[patch_to_wall_ids, :]
    return (patches_points, patches_normal, n_patches, patch_to_wall_ids)


@numba.njit()
def total_number_of_patches(polygon_points:np.ndarray, max_size: float):
    """Calculate the total number of patches.

    Parameters
    ----------
    polygon_points : np.ndarray
        points of the polygon of shape (4, 3)
    max_size : float
        maximal patch size in meters

    Returns
    -------
    n_patches : int
        number of patches

    """
    size = np.empty(polygon_points.shape[1])
    for i in range(polygon_points.shape[1]):
        size[i] = polygon_points[:, i].max() - polygon_points[:, i].min()
    patch_nums = np.array([int(n) for n in size/max_size])
    if patch_nums[2] == 0:
        x_idx = 0
        y_idx = 1
    if patch_nums[1] == 0:
        x_idx = 0
        y_idx = 2
    if patch_nums[0] == 0:
        x_idx = 1
        y_idx = 2

    return patch_nums[x_idx]*patch_nums[y_idx]
