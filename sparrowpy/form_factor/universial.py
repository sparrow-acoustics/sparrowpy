"""methods for universal form factor calculation."""
import numpy as np
import sparrowpy.geometry as geom
from sparrowpy.form_factor import integration
try:
    import numba
    prange = numba.prange
except ImportError:
    numba = None
    prange = range

def patch2patch_ff_universal(patches_points: np.ndarray,
                             patches_normals: np.ndarray,
                             patches_areas: np.ndarray,
                             visible_patches:np.ndarray):
    """Calculate the form factors between patches (universal method).

    This method computes form factors between polygonal patches via numerical
    integration. Support is only guaranteed for convex polygons.
    Because this function relies on numpy, all patches must have
    the same number of sides.

    Parameters
    ----------
    patches_points : np.ndarray
        vertices of all patches of shape (n_patches, n_vertices, 3)
    patches_normals : np.ndarray
        normal vectors of all patches of shape (n_patches, 3)
    patches_areas : np.ndarray
        areas of all patches of shape (n_patches)
    visible_patches : np.ndarray
        index list of all visible patches combinations (n_combinations, 2)

    Returns
    -------
    form_factors : np.ndarray
        form factors between all patches of shape (n_patches, n_patches)
        note that just i_source < i_receiver are calculated since
        patches_areas[i] * ff[i, j] = patches_areas[j] * ff[j, i]

    """
    n_patches = patches_areas.shape[0]
    form_factors = np.zeros((n_patches, n_patches))

    for visID in prange(visible_patches.shape[0]):
        i = int(visible_patches[visID, 0])
        j = int(visible_patches[visID, 1])
        form_factors[i,j] = calc_form_factor(
                    patches_points[i], patches_normals[i], patches_areas[i],
                    patches_points[j], patches_normals[j])

    return form_factors

def calc_form_factor(source_pts: np.ndarray, source_normal: np.ndarray,
                     source_area: np.ndarray, receiver_pts: np.ndarray,
                     receiver_normal: np.ndarray,
                     ) -> float:
    """Return the form factor based on input patches geometry.

    Parameters
    ----------
    receiver_pts: np.ndarray
        receiver patch vertex coordinates (n_vertices,3)

    receiver_normal: np.ndarray
        receiver patch normal (3,)

    receiver_area: float
        receiver patch area

    source_pts: np.ndarray
        source patch vertex coordinates (n_vertices,3)

    source_normal: np.ndarray
        source patch normal (3,)

    source_area: float
        source patch area

    Returns
    -------
    form_factor: float
        form factor

    """
    if geom._coincidence_check(receiver_pts, source_pts):
        form_factor = integration.nusselt_integration(
                    patch_i=source_pts, patch_i_normal=source_normal,
                    patch_j=receiver_pts, patch_j_normal=receiver_normal,
                    nsamples=64)
    else:
        form_factor = integration.stokes_integration(patch_i=source_pts,
                                             patch_j=receiver_pts,
                                             patch_i_area=source_area,
                                             approx_order=4)

    return form_factor

## accumulation of propagation effects (air attenuation, scattering)

def _form_factors_with_directivity(
        visibility_matrix, form_factors, n_bins, patches_center,
        patches_area, air_attenuation,
        absorption, absorption_index, patch_to_wall_ids,
        scattering, scattering_index, sources, receivers):
    """Calculate the form factors with directivity."""
    n_patches = patches_center.shape[0]
    form_factors_tilde = np.zeros((n_patches, n_patches, n_patches, n_bins))
    # loop over previous patches, current and next patch

    for ii in prange(n_patches**3):
        h = ii % n_patches
        i = int(ii/n_patches) % n_patches
        j = int(ii/n_patches**2) % n_patches
        visible_hi = visibility_matrix[
            h, i] if h < i else visibility_matrix[i, h]
        visible_ij = visibility_matrix[
            i, j] if i < j else visibility_matrix[j, i]
        if visible_hi and visible_ij:
            difference_receiver = patches_center[i]-patches_center[j]
            wall_id_i = int(patch_to_wall_ids[i])
            difference_receiver /= np.linalg.norm(difference_receiver)
            ff = form_factors[i, j] if i<j else (form_factors[j, i]
                                                 *patches_area[i]/patches_area[j])

            distance = np.linalg.norm(difference_receiver)
            form_factors_tilde[h, i, j, :] = ff
            if air_attenuation is not None:
                form_factors_tilde[h, i, j, :] = form_factors_tilde[
                    h, i, j, :] * np.exp(-air_attenuation * distance)

            if scattering is not None:
                scattering_factor = get_scattering_data(
                    patches_center[h], patches_center[i], patches_center[j],
                    sources, receivers, wall_id_i,
                    scattering, scattering_index)
                form_factors_tilde[h, i, j, :] = form_factors_tilde[
                    h, i, j, :] * scattering_factor

            if absorption is not None:
                source_wall_idx = absorption_index[wall_id_i]
                form_factors_tilde[h, i, j, :] = form_factors_tilde[
                    h, i, j, :] * (1-absorption[source_wall_idx])

    return form_factors_tilde


def _form_factors_with_directivity_dim(
        visibility_matrix, form_factors, n_bins, patches_center,
        patches_area,
        air_attenuation,
        patch_to_wall_ids,
        scattering, scattering_index, sources, receivers):
    """Calculate the form factors with directivity."""
    n_patches = patches_center.shape[0]
    n_directions = receivers.shape[1] if receivers is not None else 1
    form_factors_tilde = np.zeros((n_patches, n_patches, n_directions, n_bins))
    # loop over previous patches, current and next patch

    for ii in prange(n_patches**2):
        i = ii % n_patches
        j = int(ii/n_patches) % n_patches
        visible_ij = visibility_matrix[
            i, j] if i < j else visibility_matrix[j, i]
        if visible_ij:
            difference_receiver = patches_center[i]-patches_center[j]
            wall_id_i = int(patch_to_wall_ids[i])
            difference_receiver /= np.linalg.norm(difference_receiver)
            ff = form_factors[i, j] if i<j else (form_factors[j, i]
                                                 *patches_area[j]/patches_area[i])

            distance = np.linalg.norm(difference_receiver)
            form_factors_tilde[i, j, :, :] = ff
            if air_attenuation is not None:
                form_factors_tilde[i, j, :, :] = form_factors_tilde[
                    i, j, :, :] * np.exp(-air_attenuation * distance)

            if scattering is not None:
                scattering_factor = get_scattering_data_source(
                    patches_center[i], patches_center[j],
                    sources, wall_id_i,
                    scattering, scattering_index)
                form_factors_tilde[i, j, :, :] = form_factors_tilde[
                    i, j, :, :] * scattering_factor

    return form_factors_tilde


## scattering data access

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

    for i in range(n_patches):
        difference_receiver = pos_i[i]-pos_j
        difference_receiver /= np.linalg.norm(
            difference_receiver)
        receiver_idx[i] = np.argmin(np.sum(
            (receivers[wall_id_i[i], :]-difference_receiver)**2, axis=-1),
            axis=-1)


    return receiver_idx


def get_scattering_data(
        pos_h:np.ndarray, pos_i:np.ndarray, pos_j:np.ndarray,
        sources:np.ndarray, receivers:np.ndarray, wall_id_i:np.ndarray,
        scattering:np.ndarray, scattering_index:np.ndarray):
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
    difference_source = pos_h-pos_i
    difference_receiver = pos_i-pos_j

    difference_source /= np.linalg.norm(difference_source)
    difference_receiver /= np.linalg.norm(difference_receiver)
    source_idx = np.argmin(np.sum(
        (sources[wall_id_i, :, :]-difference_source)**2, axis=-1))
    receiver_idx = np.argmin(np.sum(
        (receivers[wall_id_i, :]-difference_receiver)**2, axis=-1))
    return scattering[scattering_index[wall_id_i],
        source_idx, receiver_idx, :]


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
    difference_source = pos_h-pos_i
    difference_source /= np.linalg.norm(difference_source)
    source_idx = np.argmin(np.sum(
        (sources[wall_id_i, :, :]-difference_source)**2, axis=-1))
    return scattering[scattering_index[wall_id_i], source_idx]


if numba is not None:
    patch2patch_ff_universal = numba.njit(parallel=True)(
        patch2patch_ff_universal)
    calc_form_factor = numba.njit()(calc_form_factor)
    _form_factors_with_directivity_dim = numba.njit(parallel=True)(
        _form_factors_with_directivity_dim)
    _form_factors_with_directivity = numba.njit(parallel=True)(
        _form_factors_with_directivity)
    get_scattering_data_receiver_index = numba.njit()(
        get_scattering_data_receiver_index)
    get_scattering_data = numba.njit()(get_scattering_data)
    get_scattering_data_source = numba.njit()(get_scattering_data_source)


