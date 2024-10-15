"""Implementation of the receiver energy calculation."""
import numba
import numpy as np
from sparapy.radiosity_fast.universal_ff.univ_form_factor import pt_solution

@numba.njit()
def _kang(
        patch_receiver_distance, patches_normal, n_bins):
    receiver_factor = np.empty((
        patch_receiver_distance.shape[0], n_bins))
    for i in range(patch_receiver_distance.shape[0]):
        R = np.sqrt(np.sum((patch_receiver_distance[i, :]**2)))

        cos_xi = np.abs(np.sum(
            patches_normal[i, :]*np.abs(patch_receiver_distance[i, :]))) / R

        # Equation 20
        receiver_factor[i, :] = cos_xi / (np.pi * R**2)
    return receiver_factor

#@numba.njit(parallel=True)
def _universal(
        receiver_pos, patches_points, n_bins):
    
    receiver_factor = np.empty((patches_points.shape[0], n_bins))
    

    for i in numba.prange(patches_points.shape[0]):
        receiver_factor[i, :] = pt_solution(point=receiver_pos,
                        patch_points=patches_points[i,:], mode="receiver")

    return receiver_factor
