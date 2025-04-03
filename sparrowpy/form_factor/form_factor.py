"""Form factor calculation for radiosity."""
try:
    import numba
    prange = numba.prange
except ImportError:
    numba = None
    prange = range
import numpy as np
from sparrowpy.radiosity_fast import geometry
from .universal import ( calc_form_factor )


def kang(
        patches_center:np.ndarray, patches_normal:np.ndarray,
        patches_size:np.ndarray, visible_patches:np.ndarray) -> np.ndarray:
    """Calculate the form factors between patches.

    Parameters
    ----------
    patches_center : np.ndarray
        center points of all patches of shape (n_patches, 3)
    patches_normal : np.ndarray
        normal vectors of all patches of shape (n_patches, 3)
    patches_size : np.ndarray
        size of all patches of shape (n_patches, 3)
    visible_patches : np.ndarray
        index list of all visible patches combinations (n_combinations, 2)

    Returns
    -------
    form_factors : np.ndarray
        form factors between all patches of shape (n_patches, n_patches)
        note that just i_source < i_receiver are calculated ff[i, j] = ff[j, i]

    """
    n_patches = patches_center.shape[0]
    form_factors = np.zeros((n_patches, n_patches))
    for i in prange(visible_patches.shape[0]):
        i_source = int(visible_patches[i, 0])
        i_receiver = int(visible_patches[i, 1])
        source_center = patches_center[i_source]
        source_normal = patches_normal[i_source]
        receiver_center = patches_center[i_receiver]
        # calculation of form factors
        receiver_normal = patches_normal[i_receiver]
        dot_product = np.dot(receiver_normal, source_normal)

        if dot_product == 0:  # orthogonal

            if np.abs(source_normal[0]) > 1e-5:
                idx_source = {2, 1}
                dl = source_center[2]
                dm = source_center[1]
                dd_l = patches_size[i_source, 2]
                dd_m = patches_size[i_source, 1]
            elif np.abs(source_normal[1]) > 1e-5:
                idx_source = {2, 0}
                dl = source_center[2]
                dm = source_center[0]
                dd_l = patches_size[i_source, 2]
                dd_m = patches_size[i_source, 0]
            elif np.abs(source_normal[2]) > 1e-5:
                idx_source = {0, 1}
                dl = source_center[1]
                dm = source_center[0]
                dd_l = patches_size[i_source, 1]
                dd_m = patches_size[i_source, 0]

            if np.abs(receiver_normal[0]) > 1e-5:
                idx_l = 1 if 1 in idx_source else 2
                idx_s = 0
                idx_r = 2 if 1 in idx_source else 1
                dl_prime = receiver_center[1]
                dn_prime = receiver_center[2]
            elif np.abs(receiver_normal[1]) > 1e-5:
                idx_l = 0 if 0 in idx_source else 2
                idx_s = 1
                idx_r = 2 if 0 in idx_source else 0
                dl_prime = receiver_center[0]
                dn_prime = receiver_center[2]
            elif np.abs(receiver_normal[2]) > 1e-5:
                idx_l = 0 if 0 in idx_source else 1
                idx_s = 2
                idx_r = 1 if 0 in idx_source else 0
                dl_prime = receiver_center[1]
                dn_prime = receiver_center[0]

            dm = np.abs(
                source_center[idx_s]-receiver_center[idx_s])
            dl = source_center[idx_l]
            dl_prime = receiver_center[idx_l]
            dn_prime = np.abs(
                source_center[idx_r]-receiver_center[idx_r])

            d = np.sqrt( ( (dl - dl_prime) ** 2 ) + ( dm ** 2 ) + (
                dn_prime ** 2) )

            # Equation 13
            A = ( dm - ( 0.5 * dd_m ) ) / ( np.sqrt( ( (
                dl - dl_prime) ** 2 ) + ( ( dm - (
                    0.5 * dd_m ) ) ** 2 ) + ( dn_prime ** 2) ) )

            # Equation 14
            B_num = dm + (0.5 * dd_m)
            B_denum =  np.sqrt(( np.square(dl - dl_prime) ) + (
                np.square(dm + (0.5*dd_m)) ) + (
                    np.square(dn_prime)) )
            B = B_num/B_denum

            # Equation 15
            one = np.arctan( np.abs( ( dl - (
                0.5*dd_l) - dl_prime ) / (dn_prime) ) )
            two = np.arctan( np.abs( ( dl + (
                0.5*dd_l) - dl_prime ) / (dn_prime) ) )

            k = -1 if np.abs(dl - dl_prime) < 1e-12 else 1

            theta = np.abs( one - (k*two) )

            # Equation 11
            ff =  (
                1 / (2 * np.pi) ) * (np.abs(
                    (A ** 2) - (B ** 2) )) * theta

        else:
            # parallel
            if np.abs(receiver_normal[0]) > 1e-5:
                dl = receiver_center[1]
                dm = receiver_center[0]
                dn = receiver_center[2]
                dl_prime = source_center[1]
                dm_prime = source_center[0]
                dn_prime = source_center[2]
                dd_l = patches_size[i_source, 1]
                if patches_size.shape[1] > 2:
                    dd_n = patches_size[i_source, 2]
                else:
                    dd_n = patches_size[i_source, 1]
            elif np.abs(receiver_normal[1]) > 1e-5:
                dl = receiver_center[0]
                dm = receiver_center[1]
                dn = receiver_center[2]
                dl_prime = source_center[0]
                dm_prime = source_center[1]
                dn_prime = source_center[2]
                dd_l = patches_size[i_source, 0]
                if patches_size.shape[1] > 2:
                    dd_n = patches_size[i_source, 2]
                else:
                    dd_n = patches_size[i_source, 1]
            elif np.abs(receiver_normal[2]) > 1e-5:
                dl = receiver_center[1]
                dm = receiver_center[2]
                dn = receiver_center[0]
                dl_prime = source_center[1]
                dm_prime = source_center[2]
                dn_prime = source_center[0]
                dd_l = patches_size[i_source, 1]
                dd_n = patches_size[i_source, 0]

            d = np.sqrt(
                ( (dl - dl_prime) ** 2 ) +
                ( (dn - dn_prime) ** 2 ) +
                ( (dm - dm_prime) ** 2 ) )
            # Equation 16
            ff =  ( dd_l * dd_n * ( (
                dm-dm_prime) ** 2 ) ) / ( np.pi * ( d**4 ) )

        form_factors[i_source, i_receiver] = ff
    return form_factors


def universal(patches_points: np.ndarray, patches_normals: np.ndarray,
              patches_areas: np.ndarray, visible_patches:np.ndarray):
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
                scattering_factor = geometry.get_scattering_data(
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
        absorption, absorption_index, patch_to_wall_ids,
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
                scattering_factor = geometry.get_scattering_data_source(
                    patches_center[i], patches_center[j],
                    sources, wall_id_i,
                    scattering, scattering_index)
                form_factors_tilde[i, j, :, :] = form_factors_tilde[
                    i, j, :, :] * scattering_factor

            if absorption is not None:
                source_wall_idx = absorption_index[wall_id_i]
                form_factors_tilde[i, j, :, :] = form_factors_tilde[
                    i, j, :, :] * (1-absorption[source_wall_idx])

    return form_factors_tilde


if numba is not None:
    _form_factors_with_directivity_dim = numba.njit(parallel=True)(
        _form_factors_with_directivity_dim)
    kang = numba.njit(parallel=True)(kang)
    universal = numba.njit(parallel=True)(universal)
    _form_factors_with_directivity = numba.njit(parallel=True)(
        _form_factors_with_directivity)
