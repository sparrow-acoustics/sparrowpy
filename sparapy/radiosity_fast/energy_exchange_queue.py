"""Recursive Energy exchange functions for the fast radiosity solver."""

import numba
import numpy as np
from . import geometry

@numba.njit(parallel=True)
def _init_energy_1(
    energy_0,
    distance_0,
    source_position: np.ndarray,
    patches_center: np.ndarray,
    visible_patches: np.ndarray,
    patches_areas: np.ndarray,
    n_bins: int,
    patch_to_wall_ids: np.ndarray,
    absorption: np.ndarray,
    absorption_index: np.ndarray,
    form_factors: np.ndarray,
    sources: np.ndarray,
    receivers: np.ndarray,
    scattering: np.ndarray,
    scattering_index: np.ndarray,
):
    energy_1 = np.empty((2*visible_patches.shape[0] , int(n_bins)))
    distance_1 = np.empty((2*visible_patches.shape[0]))
    indices = np.empty((2,2*visible_patches.shape[0]))

    for ii in numba.prange(visible_patches.shape[0]):
        for jj in range(2):
            if jj == 0:
                i = visible_patches[ii, 0]
                j = visible_patches[ii, 1]
            else:
                j = visible_patches[ii, 0]
                i = visible_patches[ii, 1]
            wall_id_i = int(patch_to_wall_ids[i])

            indices[0,2*ii+jj] = i
            indices[1,2*ii+jj] = j

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
            energy_1[2*ii+jj,:] = energy_0[i, :] * ff * absorption_factor * scattering_factor
            distance_1[2*ii+jj] = distance_0[i] + distance

    return (indices, energy_1, distance_1)


@numba.njit(parallel=True)
def _calculate_energy_exchange_first_order(
    ir,
    energy_0,
    distance_0,
    energy_1,
    distance_1,
    indices,
    patch_receiver_distance,
    patch_receiver_energy,
    speed_of_sound,
    histogram_time_resolution,
    n_bins,
    thres=1e-6,
):
    energy_11   = np.zeros((indices.shape[1],n_bins))
    distance_11 = np.zeros((indices.shape[1],))

    energy_01 = energy_0*patch_receiver_energy
    distance_01 = distance_0+patch_receiver_distance

    for i in numba.prange(indices.shape[1]):
        ii = numba.int64(indices[0,i])
        energy_11[i] = energy_1[i]*patch_receiver_energy[ii]
        distance_11[i] = distance_1[i]+patch_receiver_distance[ii]

    for i_freq in numba.prange(int(n_bins)):
        i0 = np.nonzero(energy_01[:,i_freq] > thres)[0]
        ir[:,i_freq] = _collect_receiver_from_queue(
            ir[:,i_freq],
            energy_01[i0,i_freq],
            distance_01[i0],
            speed_of_sound,
            histogram_time_resolution,
        )
        i1 = np.nonzero(energy_11[:,i_freq] > thres)[0]
        ir[:,i_freq] = _collect_receiver_from_queue(
            ir[:,i_freq],
            energy_11[i1,i_freq],
            distance_11[i1],
            speed_of_sound,
            histogram_time_resolution,
        )

    return ir


@numba.njit(parallel=True)
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
    queue[:,0:2] = np.transpose(indices)

    for ii in numba.prange(queue.shape[0]):
        queue[ii,-1] = distance_0[int(queue[ii,0])]

    for i_freq in numba.prange(energy_0.shape[-1]):
        queue[:,2] = energy_0[:,i_freq]
        ir[:,i_freq] = _energy_exchange(
            ir[:,i_freq],
            queue,
            form_factors_tilde,
            visibility_matrix,
            distance_i_j,
            patch_receiver_energy[:,i_freq],
            patch_receiver_distance,
            i_freq,
            speed_of_sound,
            histogram_time_resolution,
            threshold,
        )

    return ir

@numba.njit()
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
    n_vals = queue.shape[1]
    queue=queue.reshape(n_vals*queue.shape[0],)

    while queue.shape[0] > 0:

        row = queue[0:n_vals]
        queue = np.delete(queue,np.arange(n_vals))

        appendix, ir = _shoot(
        row,
        form_factors_tilde,
        patch_distances,
        patch_receiver_energy,
        patch_receiver_distance,
        ir,
        i_freq,
        visibility_matrix,
        speed_of_sound,
        histogram_time_resolution,
        thres
        )

        queue = np.append(queue, appendix)

    return ir

@numba.njit()
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
    dj = d + patch_distances[i, j]

    eR = ej * patch_receiver_energy[j]
    dR = dj + patch_receiver_distance[j]

    jj = np.nonzero(eR > thres)[0]
    
    ir = _collect_receiver_from_queue(
        ir,
        eR[jj],
        dR[jj],
        speed_of_sound,
        histogram_time_resolution,
    )

    jjj = np.nonzero(j)[0][jj]
    queue_out = np.empty((jj.shape[0], 4))

    queue_out[:, 0] = i * np.ones_like(jjj)
    queue_out[:, 1] = jjj
    queue_out[:, 2] = ej[jj]
    queue_out[:, 3] = dj[jj]

    return queue_out.reshape(jjj.shape[0]*4,), ir

@numba.njit()
def _collect_receiver_from_queue(
    ir,
    energy,
    distance,
    speed_of_sound,
    histogram_time_resolution,
):
    samples_delay = np.zeros_like(distance,dtype=np.int64)

    for i in range(distance.shape[0]):
        samples_delay[i] = np.floor(distance[i] / speed_of_sound / histogram_time_resolution)

    sampleIDs = np.nonzero(samples_delay < ir.shape[0])[0]

    for sample in sampleIDs:
        ir[samples_delay[sample]] += energy[sample]

    return ir
