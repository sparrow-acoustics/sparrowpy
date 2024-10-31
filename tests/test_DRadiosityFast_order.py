"""Test radiosity module."""
import numpy as np
import numpy.testing as npt
import pytest
import os
import pyfar as pf

import sparapy as sp


create_reference_files = False

@pytest.mark.parametrize('walls', [
    # perpendicular walls
    [0, 2], [0, 3], [0, 4], [0, 5],
    [1, 2], [1, 3], [1, 4], [1, 5],
    [2, 0], [2, 1], [2, 4], [2, 5],
    [3, 0], [3, 1], [3, 4], [3, 5],
    [4, 0], [4, 1], [4, 2], [4, 3],
    [5, 0], [5, 1], [5, 2], [5, 3],
    # parallel walls
    [0, 1], [2, 3], [4, 5],
    [1, 0], [3, 2], [5, 4],
    ])
@pytest.mark.parametrize('patch_size', [
    0.5,
    ])
def test_calc_form_factor_perpendicular_distance(
        sample_walls, walls, patch_size):
    """Test form factor calculation for perpendicular walls."""
    wall_source = sample_walls[walls[0]]
    wall_receiver = sample_walls[walls[1]]
    patch_1 = sp.radiosity.Patches(wall_source, patch_size, [1], 0)
    patch_2 = sp.radiosity.Patches(wall_receiver, patch_size, [0], 1)
    patches = [patch_1, patch_2]
    patch_1.calculate_form_factor(patches)
    patch_2.calculate_form_factor(patches)

    radiosity = sp.DRadiosityFast.from_polygon(
        [wall_source, wall_receiver], patch_size)
    radiosity.bake_geometry(algorithm='order')

    patch_pos = np.array([patch.center for patch in patch_1.patches])
    if (np.abs(patch_pos- radiosity.patches_center[:4, :])<1e-5).all():
        npt.assert_almost_equal(
            radiosity.form_factors[:4, 4:], patch_1.form_factors)
    else:
        npt.assert_almost_equal(
            radiosity.form_factors[:4, 4:], patch_1.form_factors.T)

    patch_pos = np.array([patch.center for patch in patch_2.patches])
    if (np.abs(patch_pos- radiosity.patches_center[4:, :])<1e-5).all():
        npt.assert_almost_equal(radiosity.form_factors[4:, :4], 0)
    else:
        npt.assert_almost_equal(radiosity.form_factors[4:, :4], 0)


@pytest.mark.parametrize('walls', [
    # perpendicular walls
    [0, 2],
    # parallel walls
    [0, 1],
    ])
@pytest.mark.parametrize('patch_size', [
    0.5,
    ])
def test_form_factors_directivity_for_diffuse(
        sample_walls, walls, patch_size, sofa_data_diffuse):
    """Test form factor calculation for perpendicular walls."""
    wall_source = sample_walls[walls[0]]
    wall_receiver = sample_walls[walls[1]]
    walls = [wall_source, wall_receiver]

    radiosity = sp.DRadiosityFast.from_polygon(
        walls, patch_size)
    data, sources, receivers = sofa_data_diffuse
    radiosity.set_wall_scattering(
        np.arange(len(walls)), data, sources, receivers)
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.set_wall_absorption(
        np.arange(len(walls)),
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.bake_geometry(algorithm='order')
    npt.assert_almost_equal(radiosity._form_factors_tilde.shape, (8, 8, 4, 4))
    # test _form_factors_tilde
    for i in range(radiosity._form_factors_tilde.shape[0]):
        for j in range(radiosity._form_factors_tilde.shape[0]):
            if i < j:
                npt.assert_almost_equal(
                    radiosity._form_factors_tilde[i, j, :, :],
                    radiosity._form_factors[i, j])
            else:
                npt.assert_almost_equal(
                    radiosity._form_factors_tilde[i, j, :, :],
                    radiosity._form_factors[j, i])


@pytest.mark.parametrize('patch_size', [
    0.5,
    1,
    ])
@pytest.mark.parametrize('source_pos', [
    np.array([.2, .2, .2]),
    # np.array([20, 2, 2]),
    ])
@pytest.mark.parametrize('receiver_pos', [
    np.array([.3, .4, .5]),
    # np.array([3, 40, 2]),
    ])
@pytest.mark.parametrize('max_order_k', [
    0, 1, 2
    ])
def test_order_vs_old_implementation(
        patch_size, source_pos, receiver_pos, max_order_k,
        sofa_data_diffuse):
    # note that order k=0 means one reflection and k=1 means two reflections
    # (2nd order)
    data, sources, receivers = sofa_data_diffuse
    walls = [0, 1]

    # source_pos = np.array([0.3, 0.5, 0.5])
    # receiver_pos = np.array([0.5, 0.5, 0.5])
    walls = sp.testing.shoebox_room_stub(5, 6, 4)
    walls = sp.testing.shoebox_room_stub(1, 1, 1)
    walls = walls[0:3:2]
    length_histogram = 0.5
    time_resolution = 1e-3
    speed_of_sound = 346.18

    radiosity_old = sp.radiosity.Radiosity(
        walls, patch_size, max_order_k, length_histogram,
        speed_of_sound=speed_of_sound,
        sampling_rate=1/time_resolution, absorption=0)

    radiosity_old.run(
        sp.geometry.SoundSource(source_pos, [1, 0, 0], [0, 0, 1]))
    histogram_old = radiosity_old.energy_at_receiver(
        sp.geometry.Receiver(receiver_pos, [1, 0, 0], [0, 0, 1]),
        ignore_direct=True)

    radiosity = sp.DRadiosityFast.from_polygon(
        walls, patch_size)

    radiosity.set_wall_scattering(
        np.arange(len(walls)), data, sources, receivers)
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.set_wall_absorption(
        np.arange(len(walls)),
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.bake_geometry(algorithm='order')

    radiosity.init_source_energy(source_pos, algorithm='order')
    histogram = radiosity.calculate_energy_exchange_receiver(
        receiver_pos, speed_of_sound, time_resolution, length_histogram,
        max_depth=max_order_k, algorithm='order')

    # get E_matrix_old (n_bins, max_order_k+1, n_patches, n_samples)
    # E_matrix (n_patches, n_directions, n_bins, n_samples)
    E_matrix_old = []
    for patch in radiosity_old.patch_list:
        E_matrix_old.append(patch.E_matrix)
    E_matrix_old = np.array(E_matrix_old)
    # # test E_matrix
    # for i_bin in range(data.n_bins):
    #     i_dir = 0

    #     npt.assert_allclose(
    #         radiosity.E_matrix_total[0, i_dir, i_bin],
    #         E_matrix_old[0][0, 0, 0, :])

    # radiosity.E_matrix_total  # n_patches, n_directions, n_bins, n_samples
    # E_matrix_old  # n_patches, n_bins, max_order_k+1, n_patches,n_samples
    # npt.assert_allclose(
    #     radiosity.E_matrix_total[0, 0, 0, :],
    #     E_matrix_old[0, 0, 0, 0, :])
    # npt.assert_allclose(
    #     radiosity.E_matrix_total[1, 0, 0, :],
    #     E_matrix_old[0, 0, 1, 0, :])

    # compare histogram
    for i in range(4):
        assert np.sum(histogram[i, :])>0
        npt.assert_allclose(
            np.sum(histogram[i, :]), np.sum(histogram_old[0, :]),
            err_msg=f'histogram i_bin={i}', rtol=3e-3)
        # npt.assert_almost_equal( histogram[0, :], histogram_old[0, :])
        # npt.assert_almost_equal(
        #     histogram[0, histogram[0, :]>0],
        #     histogram_old[0, histogram_old[0, :]>0])


@pytest.mark.parametrize('patch_size', [1, 0.5])
def test_order_reference(
        patch_size, sample_walls, sofa_data_diffuse):
    """Test if the results changes."""
    # note that order k=0 means one reflection and k=1 means two reflections
    # (2nd order)
    data, sources, receivers = sofa_data_diffuse
    data = pf.FrequencyData(data.freq[..., :1], data.frequencies[0])

    source_pos = np.array([0.3, 0.5, 0.5])
    receiver_pos = np.array([0.5, 0.5, 0.5])
    length_histogram = 0.1
    time_resolution = 1e-3
    speed_of_sound = 346.18

    radiosity = sp.DRadiosityFast.from_polygon(
        sample_walls, patch_size)

    radiosity.set_wall_scattering(
        np.arange(len(sample_walls)), data, sources, receivers)
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.set_wall_absorption(
        np.arange(len(sample_walls)),
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.bake_geometry(algorithm='order')

    radiosity.init_source_energy(source_pos, algorithm='order')
    radiosity.calculate_energy_exchange(
        receiver_pos, speed_of_sound, time_resolution, length_histogram,
        algorithm='order', max_depth=3)

    patches_hist = radiosity.collect_receiver_energy(
        receiver_pos, speed_of_sound, time_resolution, propagation_fx=True
    )

    histogram = np.sum(patches_hist[0],axis=0)

    signal = pf.Signal(histogram, 1/time_resolution)
    signal.time /= np.max(np.abs(signal.time))

    test_path = os.path.join(
        os.path.dirname(__file__), 'test_data')

    reference_path = os.path.join(
            test_path,
            f'sim_order_{patch_size}.far')
    if create_reference_files:
        pf.io.write(reference_path, signal=signal)
    result = pf.io.read(reference_path)

    for i_frequency in range(signal.time.shape[0]):
        npt.assert_almost_equal(
            result['signal'].time[0, ...], signal.time[i_frequency, ...],
            decimal=4)
    npt.assert_almost_equal(result['signal'].times, signal.times)


@pytest.mark.parametrize('patch_size', [1])
def test_order_vs_analytic(patch_size):
    """Test vs a perfect diffuse room, after Kuttruff."""
    # note that order k=0 means one reflection and k=1 means two reflections
    # (2nd order)
    X = 5
    Y = 6
    Z = 4
    length_histogram = 0.2
    time_resolution = 1e-3
    max_order_k = 3
    speed_of_sound = 346.18
    receiver_pos = np.array([3, 4, 2])
    source_pos = np.array([2, 2, 2])

    absorption = 0.1

    sources = pf.Coordinates(0, 0, 1)
    receivers = pf.Coordinates(0, 0, 1)
    frequencies = np.array([500])
    data_scattering = pf.FrequencyData(
        np.ones((sources.csize, receivers.csize, frequencies.size)),
        frequencies)
    walls = sp.testing.shoebox_room_stub(X, Y, Z)

    radiosity = sp.DRadiosityFast.from_polygon(
        walls, patch_size)

    radiosity.set_wall_scattering(
        np.arange(len(walls)), data_scattering, sources, receivers)
    radiosity.set_air_attenuation(
        pf.FrequencyData(
            np.zeros_like(data_scattering.frequencies),
            data_scattering.frequencies))
    radiosity.set_wall_absorption(
        np.arange(len(walls)),
        pf.FrequencyData(
            np.zeros_like(data_scattering.frequencies)+absorption,
            data_scattering.frequencies))
    radiosity.bake_geometry(algorithm='order')


    radiosity.init_source_energy(source_pos, algorithm='order')
    histogram = radiosity.calculate_energy_exchange_receiver(
        receiver_pos, speed_of_sound=speed_of_sound,
        histogram_time_resolution=time_resolution,
        histogram_length=length_histogram,
        algorithm='order', max_depth=max_order_k, recalculate=True)

    reverberation_order = pf.Signal(histogram, sampling_rate=1/time_resolution)
    # desired
    S = (2*X*Y) + (2*X*Z) + (2*Y*Z)
    A = S*absorption
    alpha_dash = A/S
    V = X*Y*Z
    E_reverb_analytical = 4 / A
    w_0 = E_reverb_analytical/ V # Kuttruff Eq 4.7
    t_0 = 0.03
    t = reverberation_order.times
    # Kuttruff Eq 4.10
    reverberation_analytic = w_0 * np.exp(+(
        speed_of_sound*S*np.log(1-alpha_dash)/(4*V))*(t-t_0))

    # compare histogram
    for i in range(histogram.shape[0]):
        assert np.sum(histogram[i, :])>0
        samples_reverb_start = int(0.025/time_resolution)
        samples_reverb_end = int(0.035/time_resolution)
        npt.assert_allclose(
            10*np.log10(histogram[0, samples_reverb_start:samples_reverb_end]),
            10*np.log10(reverberation_analytic[samples_reverb_start:samples_reverb_end]),
            atol=0.3)
        # npt.assert_almost_equal(
        # histogram[0, histogram[0,:]>0],
        # histogram_old[0, histogram_old[0,:]>0])

