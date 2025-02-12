"""Recursive Energy exchange functions for the fast radiosity solver."""
import numba
import numpy as np
from . import geometry


@numba.njit(parallel=True)
def _add_directional(
        energy_0, source_position: np.ndarray,
        patches_center: np.ndarray, n_bins:float, patch_to_wall_ids:np.ndarray,
        absorption:np.ndarray, absorption_index:np.ndarray,
        sources: np.ndarray, receivers: np.ndarray,
        scattering: np.ndarray, scattering_index: np.ndarray):
    """Add scattering and absorption to the initial energy from the source.

    Parameters
    ----------
    energy_0 : np.ndarray
        energy of all patches of shape (n_patches, n_bins)
    source_position : np.ndarray
        source position of shape (3,)
    patches_center : np.ndarray
        center of all patches of shape (n_patches, 3)
    n_bins : float
        number of frequency bins.
    patch_to_wall_ids : np.ndarray
        indexes from each patch to the wall of shape (n_patches)
    absorption : np.ndarray
        absorption factor of shape (n_walls, n_bins)
    absorption_index : np.ndarray
        mapping from the wall id to absorption database index (n_walls)
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
        energy of all patches of shape (n_patches, n_directions, n_bins)

    """
    n_patches = patches_center.shape[0]
    n_directions = receivers.shape[1]
    energy_0_directivity = np.zeros((n_patches, n_directions, n_bins))
    for i in numba.prange(n_patches):
        wall_id_i = int(patch_to_wall_ids[i])
        scattering_factor = geometry.get_scattering_data_source(
            source_position, patches_center[i],
            sources, wall_id_i, scattering, scattering_index)

        absorption_factor = 1-absorption[absorption_index[wall_id_i], :]
        energy_0_directivity[i, :, :] = energy_0[i] \
            * absorption_factor * scattering_factor

    return energy_0_directivity


@numba.njit()
def energy_exchange_init_energy(
        n_samples, energy_0_directivity, distance_0,
        speed_of_sound, histogram_time_resolution):
    """Calculate energy exchange between patches.

    Parameters
    ----------
    n_samples : int
        number of samples of the histogram.
    energy_0_directivity : np.ndarray
        init energy of all patches of shape (n_patches, n_directions, n_bins)
    distance_0 : np.ndarray
        distance from the source to all patches of shape (n_patches)
    speed_of_sound : float
        speed of sound in m/s.
    histogram_time_resolution : float
        time resolution of the histogram in s.

    Returns
    -------
    E_matrix_total : np.ndarray
        energy of all patches of shape
        (n_patches, n_directions, n_bins, n_samples)

    """
    n_patches = energy_0_directivity.shape[0]
    n_directions = energy_0_directivity.shape[1]
    n_bins = energy_0_directivity.shape[2]
    E_matrix_total = np.zeros((n_patches, n_directions, n_bins, n_samples))
    for i in numba.prange(n_patches):
        n_delay_samples = int(
            distance_0[i]/speed_of_sound/histogram_time_resolution)
        E_matrix_total[i, :, :, n_delay_samples] += energy_0_directivity[i]
    return E_matrix_total


@numba.njit()
def energy_exchange(
        n_samples, energy_0_directivity, distance_0, distance_ij,
        form_factors_tilde,
        speed_of_sound, histogram_time_resolution, max_order, visible_patches):
    """Calculate energy exchange between patches.

    Parameters
    ----------
    n_samples : int
        number of samples of the histogram.
    energy_0_directivity : np.ndarray
        init energy of all patches of shape (n_patches, n_directions, n_bins)
    distance_0 : np.ndarray
        distance from the source to all patches of shape (n_patches)
    distance_ij : np.ndarray
        distance between all patches of shape (n_patches, n_patches)
    form_factors_tilde : np.ndarray
        form factors of shape (n_patches, n_patches, n_directions, n_bins)
    speed_of_sound : float
        speed of sound in m/s.
    histogram_time_resolution : float
        time resolution of the histogram in s.
    max_order : int
        maximum order of reflections.
    visible_patches : np.ndarray
        indexes of all visible patches of shape (n_visible, 2)

    Returns
    -------
    E_matrix_total : np.ndarray
        energy of all patches of shape
        (n_patches, n_directions, n_bins, n_samples)

    """
    n_patches = form_factors_tilde.shape[0]
    n_directions = form_factors_tilde.shape[2]
    n_bins = energy_0_directivity.shape[2]
    form_factors_tilde = form_factors_tilde[..., np.newaxis]
    E_matrix_total  = energy_exchange_init_energy(
        n_samples, energy_0_directivity, distance_0, speed_of_sound,
        histogram_time_resolution)
    E_matrix = np.zeros((2, n_patches, n_directions, n_bins, n_samples))
    E_matrix[0] += E_matrix_total
    if max_order == 0:
        return E_matrix_total
    for k in range(max_order):
        current_index = (1+k) % 2
        E_matrix[current_index, :, :, :] = 0
        for ii in range(visible_patches.shape[0]):
            for jj in range(2):
                if jj == 0:
                    i = visible_patches[ii, 0]
                    j = visible_patches[ii, 1]
                else:
                    j = visible_patches[ii, 0]
                    i = visible_patches[ii, 1]
                n_delay_samples = int(
                    distance_ij[i, j]/speed_of_sound/histogram_time_resolution)
                if n_delay_samples > 0:
                    E_matrix[current_index, j, :, :, n_delay_samples:] += \
                        form_factors_tilde[i, j] * E_matrix[
                            current_index-1, i, :, :, :-n_delay_samples]
                else:
                    E_matrix[current_index, j, :, :, :] += form_factors_tilde[
                        i, j] * E_matrix[current_index-1, i, :, :, :]
        E_matrix_total += E_matrix[current_index]
    return E_matrix_total


@numba.njit()
def _collect_receiver_energy(
        E_matrix_total, patch_receiver_distance,
        speed_of_sound, histogram_time_resolution, air_attenuation):
    """Collect the energy at the receiver.

    Parameters
    ----------
    ir : np.ndarray
        impulse response of shape (n_samples, n_bins)
    E_matrix_total : np.ndarray
        energy of all patches of shape
        (n_patches, n_directions, n_bins, n_samples)
    patch_receiver_distance : np.ndarray
        distance from the patch to the receiver of shape (n_patches)
    patch_receiver_energy : np.ndarray
        energy of the patch at the receiver of shape (n_patches, n_bins)
    speed_of_sound : float
        speed of sound in m/s.
    histogram_time_resolution : float
        time resolution of the histogram in s.
    receiver_idx : np.ndarray
        indexes of the direction from the patch towards the receiver,
        of shape (n_patches)

    Returns
    -------
    ir : np.ndarray
        impulse response of shape (n_samples, n_bins)

    """
    E_mat_out = np.zeros_like(E_matrix_total)
    n_patches = E_matrix_total.shape[0]
    n_bins = E_matrix_total.shape[1]

    for i in numba.prange(n_patches):
        n_delay_samples = int(np.ceil(
            patch_receiver_distance[i]/speed_of_sound/histogram_time_resolution))
        for j in range(n_bins):
            E_mat_out[i,j] = np.roll(
                E_matrix_total[i,j]*np.exp(-air_attenuation[j]*patch_receiver_distance[i]),
                n_delay_samples)

    return E_mat_out
