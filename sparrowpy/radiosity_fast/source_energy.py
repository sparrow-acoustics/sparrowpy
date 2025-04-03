"""Calculate initial energy from the source to the patch."""
try:
    import numba
    prange = numba.prange
except ImportError:
    numba = None
    prange = range

import numpy as np
from sparrowpy.form_factor.universal import (
    pt_solution as patch2point)


def _init_energy_kang(
        source_position: np.ndarray, patches_center: np.ndarray,
        patches_normal: np.ndarray, air_attenuation:np.ndarray,
        patches_size: float, n_bins:float):
    """Calculate the initial energy from the source.

    Parameters
    ----------
    source_position : np.ndarray
        source position of shape (3,)
    patches_center : np.ndarray
        center of all patches of shape (n_patches, 3)
    patches_normal : np.ndarray
        normal of all patches of shape (n_patches, 3)
    air_attenuation : np.ndarray
        air attenuation factor in Np/m (n_bins,)
    patches_size : float
        size of all patches of shape (n_patches, 3)
    n_bins : float
        number of frequency bins.

    Returns
    -------
    energy : np.ndarray
        energy of all patches of shape (n_patches)
    distance : np.ndarray
        corresponding distance of all patches of shape (n_patches)

    """
    n_patches = patches_center.shape[0]
    energy = np.empty((n_patches, n_bins))
    distance_out = np.empty((n_patches, ))
    for j in prange(n_patches):
        source_pos = source_position.copy()
        receiver_pos = patches_center[j, :].copy()
        receiver_normal = patches_normal[j, :].copy()
        receiver_size = patches_size[j, :].copy()

        if np.abs(receiver_normal[2]) > 0.99:
            i = 2
            indexes = [0, 1, 2]
        elif np.abs(receiver_normal[1]) > 0.99:
            indexes = [2, 0, 1]
            i = 1
        elif np.abs(receiver_normal[0]) > 0.99:
            i = 0
            indexes = [1, 2, 0]
        offset = receiver_pos[i]
        source_pos[i] = np.abs(source_pos[i] - offset)
        receiver_pos[i] = np.abs(receiver_pos[i] - offset)
        dl = receiver_pos[indexes[0]]
        dm = receiver_pos[indexes[1]]
        dn = receiver_pos[indexes[2]]
        dd_l = receiver_size[indexes[0]]
        dd_m = receiver_size[indexes[1]]
        S_x = source_pos[indexes[0]]
        S_y = source_pos[indexes[1]]
        S_z = source_pos[indexes[2]]

        half_l = dd_l/2
        half_m = dd_m/2

        sin_phi_delta = (dl + half_l - S_x)/ (np.sqrt(np.square(
            dl+half_l-S_x) + np.square(dm-S_y) + np.square(dn-S_z)))
        test1 = (dl - half_l) <= S_x
        test2 = S_x <= (dl + half_l)
        k_phi = -1 if test1 and test2 else 1

        sin_phi = k_phi * (dl - half_l - S_x) / (np.sqrt(np.square(
            dl-half_l-S_x) + np.square(dm-S_y) + np.square(dn-S_z)))
        if (sin_phi_delta-sin_phi) < 1e-11:
            sin_phi *= -1

        plus  = np.arctan(np.abs((dm+half_m-S_y)/np.abs(S_z)))
        minus = np.arctan(np.abs((dm-half_m-S_y)/np.abs(S_z)))

        test1 = (dm - half_m) <= S_y
        test2 = S_y <= (dm + half_m)
        k_beta = -1 if test1 and test2 else 1

        beta = np.abs(plus-(k_beta*minus))
        distance_out[j] = np.sqrt(
            np.square(dl-S_x) + np.square(dm-S_y) + np.square(dn-S_z))

        if air_attenuation is not None:
            energy[j, :] = (np.abs(sin_phi_delta-sin_phi) ) * beta / (
                4*np.pi) * np.exp(
                -air_attenuation * distance_out[j])
        else:
            energy[j, :] = (np.abs(sin_phi_delta-sin_phi) ) * beta / (
                4*np.pi)

    return (energy, distance_out)


def _init_energy_universal(
        source_position: np.ndarray, patches_center: np.ndarray,
        patches_points: np.ndarray, air_attenuation:np.ndarray,
        n_bins:float):
    """Calculate the initial energy from the source.

    Parameters
    ----------
    source_position : np.ndarray
        source position of shape (3,)
    patches_center : np.ndarray
        center of all patches of shape (n_patches, 3)
    patches_points : np.ndarray
        vertices of all patches of shape (n_patches, n_points,3)
    air_attenuation : np.ndarray
        air attenuation factor in Np/m (n_bins,)
    n_bins : float
        number of frequency bins.

    Returns
    -------
    energy : np.ndarray
        energy of all patches of shape (n_patches)
    distance : np.ndarray
        corresponding distance of all patches of shape (n_patches)

    """
    n_patches = patches_center.shape[0]
    energy = np.empty((n_patches, n_bins))
    distance_out = np.empty((n_patches, ))
    for j in prange(n_patches):
        source_pos = source_position.copy()
        receiver_pos = patches_center[j, :].copy()
        receiver_pts = patches_points[j, :, :].copy()

        distance_out[j] = np.linalg.norm(source_pos-receiver_pos)

        if air_attenuation is not None:
            energy[j,:] = np.exp(
                -air_attenuation*distance_out[j]) * patch2point(
                    point=source_pos, patch_points=receiver_pts, mode="source")
        else:
            energy[j,:] = patch2point(
                point=source_pos, patch_points=receiver_pts, mode="source")

    return (energy, distance_out)


if numba is not None:
    _init_energy_universal = numba.njit(parallel=True)(_init_energy_universal)
    _init_energy_kang = numba.njit(parallel=True)(_init_energy_kang)
