"""Recursive Energy exchange functions for the fast radiosity solver."""
import numba
import numpy as np
from . import geometry


@numba.njit(parallel=True)
def _init_energy_1(
        energy_0, distance_0, source_position: np.ndarray,
        patches_center: np.ndarray, visible_patches: np.ndarray,
        air_attenuation:np.ndarray, n_bins:float, patch_to_wall_ids:np.ndarray,
        absorption:np.ndarray, absorption_index:np.ndarray,
        form_factors: np.ndarray, sources: np.ndarray, receivers: np.ndarray,
        scattering: np.ndarray, scattering_index: np.ndarray):
    """Calculate the initial energy from the source.

    Parameters
    ----------
    energy_0 : np.ndarray
        energy of all patches of shape (n_patches)
    distance_0 : np.ndarray
        corresponding distance of all patches of shape (n_patches)
    source_position : np.ndarray
        source position of shape (3,)
    patches_center : np.ndarray
        center of all patches of shape (n_patches, 3)
    visible_patches : np.ndarray
        index list of all visible patches combinations (n_combinations, 2)
    air_attenuation : np.ndarray
        air attenuation factor in Np/m (n_bins,)
    n_bins : float
        number of frequency bins.
    patch_to_wall_ids : np.ndarray
        indexes from each patch to the wall of shape (n_patches)
    absorption : np.ndarray
        absorption factor of shape (n_walls, n_bins)
    absorption_index : np.ndarray
        mapping from the wall id to absorption database index (n_walls)
    form_factors : np.ndarray
        form factors between all patches of shape (n_patches, n_patches)
    sources : np.ndarray
        source positions of shape (n_walls, n_sources, 3)
    receivers : np.ndarray
        receiver positions of shape (n_walls, n_receivers, 3)
    scattering : np.ndarray
        scattering data of shape (n_walls, n_sources, n_receivers, n_bins)
    scattering_index : np.ndarray
        mapping from the wall id to scattering database index (n_walls)

    Returns
    -------
    energy : np.ndarray
        energy of all patches of shape (n_patches)
    distance : np.ndarray
        corresponding distance of all patches of shape (n_patches)

    """
    n_patches = patches_center.shape[0]
    energy_1 = np.zeros((n_patches, n_patches, n_bins))
    distance_1 = np.zeros((n_patches, n_patches))
    for ii in numba.prange(visible_patches.shape[0]):
        for jj in range(2):
            if jj == 0:
                i = visible_patches[ii, 0]
                j = visible_patches[ii, 1]
            else:
                j = visible_patches[ii, 0]
                i = visible_patches[ii, 1]
            wall_id_i = int(patch_to_wall_ids[i])
            scattering_factor = geometry.get_scattering_data(
                source_position, patches_center[i], patches_center[j],
                sources, receivers, wall_id_i, scattering, scattering_index)
            distance = np.linalg.norm(patches_center[i] - patches_center[j])

            ff = form_factors[i, j] if i<j else form_factors[j, i]

            absorption_factor = 1-absorption[absorption_index[wall_id_i], :]
            if air_attenuation is not None:
                energy_1[i, j, :] = energy_0[i] * ff * (
                    np.exp(-air_attenuation * distance) * absorption_factor
                    * scattering_factor)
            else:
                energy_1[i, j, :] = energy_0[i] * ff * (
                    absorption_factor * scattering_factor)

            distance_1[i, j] = distance_0[i]+distance

    return (energy_1, distance_1)



@numba.njit()
def _energy_exchange(
        ir, i_freq, h, i, energy, distance, form_factors_tilde, distance_1,
        patch_receiver_distance, patch_receiver_energy, speed_of_sound,
        histogram_time_resolution, threshold=1e-12,
        max_distance=0.1, current_depth=0, max_depth=-1):
    n_patches = form_factors_tilde.shape[0]
    energy_new = energy * form_factors_tilde[h, i, :]
    if current_depth<max_depth:
        for j in range(n_patches):
            distance_new = distance + distance_1[i, j]
            if (energy_new[j] > 0) and (distance_new < max_distance):
                # energy_new += energy * form_factors_tilde[h, i, j]
                ir = _collect_receiver_energy(
                    ir, i_freq, energy_new[j], distance_new, patch_receiver_distance[j],
                    patch_receiver_energy[j],
                    speed_of_sound, histogram_time_resolution)
                _energy_exchange(
                    ir, i_freq, i, j, energy_new[j], distance_new, form_factors_tilde,
                    distance_1, patch_receiver_distance, patch_receiver_energy,
                    speed_of_sound, histogram_time_resolution,
                    threshold, max_distance, current_depth+1, max_depth)


@numba.njit()
def _collect_receiver_energy(
        ir, i_freq,energy, distance, patch_receiver_distance, patch_receiver_energy,
        speed_of_sound, histogram_time_resolution):

    R = np.linalg.norm(patch_receiver_distance)
    d = distance+R
    samples_delay = int(d/speed_of_sound/histogram_time_resolution)
    print(f'n_old = {samples_delay}')

    # Equation 20
    ir[samples_delay,i_freq] += energy*patch_receiver_energy
    return ir



@numba.njit()
def _calculate_energy_exchange_second_order(
        ir, energy_0, distance_0, energy_1, distance_1,
        patch_receiver_distance, patch_receiver_energy ,speed_of_sound,
        histogram_time_resolution, n_patches, n_bins):
    for i_freq in range(n_bins):
        for i in range(n_patches):
            if energy_0[i, i_freq] > 0:
                ir = _collect_receiver_energy(
                    ir, i_freq, energy_0[i, i_freq],
                    distance_0[i],
                    patch_receiver_distance[i],
                    patch_receiver_energy[i, i_freq],
                    speed_of_sound, histogram_time_resolution)
            for j in range(n_patches):
                if energy_1[i, j, i_freq] > 0:
                    ir = _collect_receiver_energy(
                        ir, i_freq, energy_1[i, j, i_freq],
                        distance_1[i, j],
                        patch_receiver_distance[j],
                        patch_receiver_energy[j, i_freq],
                        speed_of_sound, histogram_time_resolution)

@numba.njit(parallel=True)
def _calculate_energy_exchange_recursive(
        ir, energy_1, distance_1,distance_i_j, form_factors_tilde,
        n_patches, patch_receiver_distance, patch_receiver_energy,
        speed_of_sound, histogram_time_resolution,
        threshold=1e-12, max_time=0.1, max_depth=-1):
    max_distance = max_time*speed_of_sound
    for i_freq in numba.prange(energy_1.shape[-1]):
        for h in range(n_patches):
            for i in range(n_patches):
                if energy_1[h, i, i_freq] > 0:
                    _energy_exchange(
                        ir, i_freq, h, i, energy_1[h, i, i_freq], distance_1[h, i],
                        form_factors_tilde[..., i_freq], distance_i_j,
                        patch_receiver_distance,
                        patch_receiver_energy[..., i_freq], speed_of_sound,
                        histogram_time_resolution,
                        threshold=threshold, max_distance=max_distance,
                        current_depth=1, max_depth=max_depth)

