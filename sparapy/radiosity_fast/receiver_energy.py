"""Implementation of the receiver energy calculation."""

import numba
import numpy as np
from .universal_ff.univ_form_factor import (
    pt_solution_receiver as patch_receiver_energies,
)


@numba.njit()
def _kang(patch_receiver_distance, patches_normal, air_attenuation):
    receiver_factor = np.empty((patch_receiver_distance.shape[0], air_attenuation.size))
    for i in range(patch_receiver_distance.shape[0]):
        R = np.sqrt(np.sum((patch_receiver_distance[i, :] ** 2)))

        cos_xi = (
            np.abs(np.sum(patches_normal[i, :] * np.abs(patch_receiver_distance[i, :])))
            / R
        )

        # Equation 20
        receiver_factor[i, :] = cos_xi * (np.exp(-air_attenuation * R)) / (np.pi * R**2)
    return receiver_factor


def _universal(
    receiver_pos: np.ndarray,
    patch_points: np.ndarray,
    patch_areas: np.ndarray,
    patch_receiver_distance: np.ndarray,
    air_attenuation: np.ndarray,
):
    receiver_factor = np.empty((patch_receiver_distance.shape[0], air_attenuation.size))

    for i in range(patch_receiver_distance.shape[0]):
        receiver_factor[i, :] = patch_receiver_energies(
            receiver_pos, patch_points[i], patch_areas[i]
        ) * np.exp(-air_attenuation * patch_receiver_distance[i])

    return receiver_factor
