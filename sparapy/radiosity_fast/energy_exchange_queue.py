"""Recursive Energy exchange functions for the fast radiosity solver."""

import numba
import numpy as np
from . import geometry


def _init_energy_1(
    energy_0,
    distance_0,
    source_position: np.ndarray,
    patches_center: np.ndarray,
    visible_patches: np.ndarray,
    patches_areas: np.ndarray,
    n_bins: float,
    patch_to_wall_ids: np.ndarray,
    absorption: np.ndarray,
    absorption_index: np.ndarray,
    form_factors: np.ndarray,
    sources: np.ndarray,
    receivers: np.ndarray,
    scattering: np.ndarray,
    scattering_index: np.ndarray,
):
    n_patches = patches_center.shape[0]
    energy_1 = np.zeros((0, n_bins))
    distance_1 = np.array([])
    hhh = np.array([])
    iii = np.array([])
    for ii in numba.prange(visible_patches.shape[0]):
        for jj in range(2):
            if jj == 0:
                i = visible_patches[ii, 0]
                j = visible_patches[ii, 1]
            else:
                j = visible_patches[ii, 0]
                i = visible_patches[ii, 1]
            wall_id_i = int(patch_to_wall_ids[i])

            hhh = np.append(hhh, i)
            iii = np.append(iii, j)

            scattering_factor = geometry.get_scattering_data(
                source_position,
                patches_center[i],
                patches_center[j],
                sources,
                receivers,
                wall_id_i,
                scattering,
                scattering_index,
            )

            distance = np.linalg.norm(patches_center[i] - patches_center[j])

            ff = (
                form_factors[i, j]
                if i < j
                else form_factors[j, i] * patches_areas[i] / patches_areas[j]
            )

            absorption_factor = 1 - absorption[absorption_index[wall_id_i], :]
            energy_1 = np.append(energy_1, [energy_0[i, :] * ff * absorption_factor * scattering_factor], axis=0)

            distance_1 = np.append(distance_1, distance_0[i] + distance)

    return (np.array([hhh, iii]), energy_1, distance_1)


def _calculate_energy_exchange_first_order(
    ir,
    energy_0,
    distance_0,
    patch_receiver_energy,
    speed_of_sound,
    histogram_time_resolution,
    n_bins,
    thres=1e-6,
):
    for i_freq in numba.prange(n_bins):
        i0 = np.nonzero(energy_0[:, i_freq] > thres)[0]
        ir = _collect_receiver_from_queue(
            ir,
            i_freq,
            energy_0[i0, i_freq],
            distance_0[i0],
            patch_receiver_energy[i0,i_freq],
            speed_of_sound,
            histogram_time_resolution,
        )


    return ir


def _calculate_energy_exchange_queue(
    ir,
    indices,
    energy_0,
    distance_0,
    distance_i_j,
    form_factors_tilde,
    visibility_matrix,
    patch_receiver_distance,
    patch_receiver_energy,
    speed_of_sound,
    histogram_time_resolution,
    threshold=1e-6,
    max_time=0.1,
    max_depth=-1,
):
    queue = np.empty((indices.shape[1], 4))
    queue[:,0:2] = np.transpose(indices).astype(int)

    for ii in range(indices.shape[1]):
        queue[ii,-1] = distance_0[int(queue[ii,0])] + distance_i_j[int(queue[ii,0]),int(queue[ii,1])]  # noqa: E501

    for i_freq in numba.prange(energy_0.shape[-1]):
        queue[:,2] = energy_0[:,i_freq]
        
        ir = _energy_exchange(
            ir,
            queue,
            form_factors_tilde,
            visibility_matrix,
            distance_i_j,
            patch_receiver_energy,
            patch_receiver_distance,
            i_freq,
            speed_of_sound,
            histogram_time_resolution,
            threshold,
        )


def _energy_exchange(
    ir,
    queue,
    form_factors_tilde,
    visibility_matrix,
    patch_distances,
    patch_receiver_energy,
    patch_receiver_distance,
    i_freq,
    speed_of_sound,
    histogram_time_resolution,
    thres=1e-6,
):
    while queue.shape[0] > 0:
        row = queue[0]
        queue = np.delete(queue, 0, 0)

        appendix, ir = _shoot(
        row,
        form_factors_tilde,
        patch_distances,
        patch_receiver_energy[:,i_freq],
        patch_receiver_distance,
        ir,
        i_freq,
        visibility_matrix,
        speed_of_sound,
        histogram_time_resolution,
        thres
        )

        queue = np.append(queue, appendix, 0)

    return ir


def _shoot(
    row_in,
    form_factors_tilde,
    patch_distances,
    patch_receiver_energy,
    patch_receiver_distance,
    ir,
    i_freq,
    visibility_matrix,
    speed_of_sound,
    histogram_time_resolution,
    thres=1e-6,
):
    h = int(row_in[0])
    i = int(row_in[1])
    e = row_in[2]
    d = row_in[3]

    j = visibility_matrix[i, :]
    j = j + visibility_matrix[:, i]  # list indices of all visible patches

    ej = e * form_factors_tilde[h, i, j, i_freq]

    eR = ej * patch_receiver_energy[j]

    jj = np.nonzero(eR > thres)[0]

    dR = d + patch_receiver_distance[jj]

    ir = _collect_receiver_from_queue(
        ir,
        i_freq,
        eR[jj],
        dR,
        patch_receiver_energy[jj],
        speed_of_sound,
        histogram_time_resolution,
    )

    queue_out = np.empty((jj.shape[0], 4))

    queue_out[:, 0] = i * np.ones_like(jj)
    queue_out[:, 1] = jj
    queue_out[:, 2] = ej[jj]
    queue_out[:, 3] = d + patch_distances[i, jj]

    return queue_out, ir

#@numba.njit()
def _collect_receiver_from_queue(
    ir,
    i_freq,
    energy,
    distance,
    patch_receiver_energy,
    speed_of_sound,
    histogram_time_resolution,
):
    samples_delay = np.zeros(distance.shape[0],dtype=int)

    for i in range(distance.shape[0]):
        samples_delay[i] = int(distance[i] / speed_of_sound / histogram_time_resolution)

    sampleIDs = np.nonzero(samples_delay < ir.shape[0])[0]

    for sample in sampleIDs:
        ir[samples_delay[sample], i_freq] += energy[sample] * patch_receiver_energy[sample]

    return ir
