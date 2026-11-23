"""Test the frequency-parallel energy exchange against the serial version."""
import numpy as np
import numpy.testing as npt
import pytest

from sparrowpy.classes.RadiosityFast import (
    _energy_exchange,
    _energy_exchange_init_energy,
)


def _energy_exchange_reference(
        n_samples, energy_0_directivity, distance_0, distance_ij,
        form_factors_tilde, patch_2_out_directions,
        speed_of_sound, histogram_time_resolution, max_order,
        visible_patches):
    """Serial implementation before the bins were parallelised."""
    n_patches = form_factors_tilde.shape[0]
    n_directions = form_factors_tilde.shape[2]
    n_bins = energy_0_directivity.shape[2]
    form_factors_tilde = form_factors_tilde[..., np.newaxis]
    E_matrix_total = _energy_exchange_init_energy(
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
                dir_id = patch_2_out_directions[i, j]
                n_delay_samples = int(
                    distance_ij[i, j]/speed_of_sound/histogram_time_resolution)
                if n_delay_samples > 0:
                    E_matrix[current_index, j, :, :, n_delay_samples:] += \
                        form_factors_tilde[i, j] * E_matrix[
                            current_index-1, i, dir_id, :, :-n_delay_samples]
                else:
                    E_matrix[current_index, j, :, :, :] += form_factors_tilde[
                        i, j] * E_matrix[current_index-1, i, dir_id, :, :]
        E_matrix_total += E_matrix[current_index]
    return E_matrix_total


def _random_inputs(n_bins, rng):
    n_patches = 6
    n_directions = 3
    n_samples = 40
    speed_of_sound = 343.
    time_resolution = 1e-3
    meter_per_sample = speed_of_sound * time_resolution

    energy_0 = rng.random((n_patches, n_directions, n_bins))
    distance_0 = rng.random(n_patches) * 10 * meter_per_sample
    distance_ij = rng.random((n_patches, n_patches)) * 20 * meter_per_sample
    distance_ij = (distance_ij + distance_ij.T) / 2
    # cover delays of zero samples and delays beyond the histogram
    distance_ij[0, 1] = distance_ij[1, 0] = 0.
    distance_ij[2, 3] = distance_ij[3, 2] = 2 * n_samples * meter_per_sample
    form_factors_tilde = rng.random(
        (n_patches, n_patches, n_directions, n_bins)) * 0.2
    patch_2_out_directions = rng.integers(
        0, n_directions, (n_patches, n_patches)).astype(np.int64)
    return (
        n_samples, energy_0, distance_0, distance_ij, form_factors_tilde,
        patch_2_out_directions, speed_of_sound, time_resolution)


@pytest.mark.parametrize("n_bins", [1, 5])
@pytest.mark.parametrize("max_order", [0, 1, 4])
def test_energy_exchange_matches_serial_reference(n_bins, max_order):
    rng = np.random.default_rng(0)
    (n_samples, energy_0, distance_0, distance_ij, form_factors_tilde,
     patch_2_out_directions, speed_of_sound,
     time_resolution) = _random_inputs(n_bins, rng)
    n_patches = energy_0.shape[0]
    visible_patches = np.array(
        [(i, j) for i in range(n_patches) for j in range(i+1, n_patches)],
        dtype=np.int32)

    args = (
        n_samples, energy_0, distance_0, distance_ij, form_factors_tilde,
        patch_2_out_directions, speed_of_sound, time_resolution, max_order,
        visible_patches)
    expected = _energy_exchange_reference(*args)
    actual = _energy_exchange(*args)

    assert actual.shape == (n_patches, energy_0.shape[1], n_bins, n_samples)
    npt.assert_array_equal(actual, expected)
