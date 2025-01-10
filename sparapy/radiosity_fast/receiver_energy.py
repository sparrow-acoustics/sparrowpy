"""Implementation of the receiver energy calculation."""
import numba
import numpy as np
from sparapy.radiosity_fast.universal_ff.univ_form_factor import pt_solution

@numba.njit(parallel=True)
def _universal(
        receiver_pos, patches_points):

    receiver_factor = np.empty((patches_points.shape[0]))


    for i in numba.prange(patches_points.shape[0]):
        receiver_factor[i] = pt_solution(point=receiver_pos,
                        patch_points=patches_points[i,:], mode="receiver")

    return receiver_factor
