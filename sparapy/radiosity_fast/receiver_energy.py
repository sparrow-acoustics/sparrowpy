"""Implementation of the receiver energy calculation."""
import numba
import numpy as np


@numba.njit()
def _kang(
        patch_receiver_distance, patches_normal, air_attenuation):
    receiver_factor = np.empty((
        patch_receiver_distance.shape[0], air_attenuation.size))
    for i in range(patch_receiver_distance.shape[0]):
        R = np.sqrt(np.sum((patch_receiver_distance[i, :]**2)))

        cos_xi = np.abs(np.sum(
            patches_normal[i, :]*np.abs(patch_receiver_distance[i, :]))) / R

        # Equation 20
        receiver_factor[i, :] = cos_xi * (np.exp(-air_attenuation*R)) / (
            np.pi * R**2)
    return receiver_factor
