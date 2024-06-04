"""Test radiosity module."""
import numpy as np
import numpy.testing as npt
import pytest
import pyfar as pf

import sparapy as sp

def test_init(sample_walls):
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(sample_walls, 0.2)
    npt.assert_almost_equal(radiosity.patches_points.shape, (150, 4, 3))
    npt.assert_almost_equal(radiosity.patches_area.shape, (150))
    npt.assert_almost_equal(radiosity.patches_center.shape, (150, 3))
    npt.assert_almost_equal(radiosity.patches_size.shape, (150, 3))
    npt.assert_almost_equal(radiosity.patches_normal.shape, (150, 3))


def test_check_visibility(sample_walls):
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(sample_walls, 0.2)
    radiosity.check_visibility()
    npt.assert_almost_equal(radiosity._visibility_matrix.shape, (150, 150))
    # npt.assert_array_equal(
    #     radiosity._visibility_matrix, radiosity._visibility_matrix.T)
    npt.assert_array_equal(radiosity._visibility_matrix[:25, :25], False)
    npt.assert_array_equal(radiosity._visibility_matrix[:25, 25:], True)
    npt.assert_array_equal(radiosity._visibility_matrix[25:50, 25:50], False)
    assert np.sum(radiosity._visibility_matrix) == 25*5*25*6/2


def test_check_visibility_wrapper(sample_walls):
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(sample_walls, 0.2)
    radiosity.check_visibility()
    visibility_matrix = sp.radiosity_fast.check_visibility(
        radiosity.patches_center, radiosity.patches_normal)
    npt.assert_almost_equal(radiosity._visibility_matrix, visibility_matrix)


def test_compute_form_factors(sample_walls):
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(sample_walls, 0.2)
    radiosity.check_visibility()
    radiosity.calculate_form_factors()
    npt.assert_almost_equal(radiosity.form_factors.shape, (150, 150))


def test_compute_form_factors_wrapper(sample_walls):
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(sample_walls, 0.2)
    radiosity.check_visibility()
    radiosity.calculate_form_factors()
    form_factors = sp.radiosity_fast.form_factor_kang(
        radiosity.patches_center, radiosity.patches_normal,
        radiosity.patches_size, radiosity.visibility_matrix)
    npt.assert_almost_equal(radiosity.form_factors, form_factors)


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

    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(
        [wall_source, wall_receiver], patch_size)
    radiosity.check_visibility()
    radiosity.calculate_form_factors()

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
    1,
    ])
def test_full(
        sample_walls, walls, patch_size, sofa_data_diffuse):
    """Test form factor calculation for perpendicular walls."""
    source_pos = np.array([0.5, 0.5, 0.5])
    receiver_pos = np.array([0.5, 0.5, 0.5])
    wall_source = sample_walls[walls[0]]
    wall_receiver = sample_walls[walls[1]]
    walls = [wall_source, wall_receiver]
    length_histogram = 1
    time_resolution = 1e-3
    k = 3
    speed_of_sound = 346.18

    radiosity_old = sp.radiosity.Radiosity(
        walls, patch_size, k, length_histogram,
        speed_of_sound=speed_of_sound,
        sampling_rate=1/time_resolution, absorption=0)
    radiosity_old.run(
        sp.geometry.SoundSource(source_pos, [1, 0, 0], [0, 0, 1]))
    histogram_old = radiosity_old.energy_at_receiver(
        sp.geometry.Receiver(receiver_pos, [1, 0, 0], [0, 0, 1]))

    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(
        walls, patch_size)
    radiosity.check_visibility()
    radiosity.calculate_form_factors()
    data, sources, receivers = sofa_data_diffuse
    radiosity.set_wall_scattering(
        np.arange(len(walls)), data, sources, receivers)
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.set_wall_absorption(
        np.arange(len(walls)),
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.calculate_form_factors_directivity()
    radiosity.calculate_energy_exchange(k)
    radiosity.init_energy(source_pos)
    histogram = radiosity.collect_energy_receiver(
        receiver_pos, speed_of_sound=speed_of_sound,
        histogram_time_resolution=time_resolution,
        histogram_time_length=length_histogram)

    for i in range(4):
        npt.assert_almost_equal(histogram[1:, i], histogram_old[0, 1:])

    npt.assert_almost_equal(
        np.sum(histogram[1:, :], axis=0),
        np.sum(histogram_old[0, 1:]))


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
def test_form_factors_directivity_for_diffuse(
        sample_walls, walls, patch_size, sofa_data_diffuse):
    """Test form factor calculation for perpendicular walls."""
    wall_source = sample_walls[walls[0]]
    wall_receiver = sample_walls[walls[1]]
    walls = [wall_source, wall_receiver]
    k = 5

    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(
        walls, patch_size)
    radiosity.check_visibility()
    radiosity.calculate_form_factors()
    data, sources, receivers = sofa_data_diffuse
    radiosity.set_wall_scattering(
        np.arange(len(walls)), data, sources, receivers)
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.set_wall_absorption(
        np.arange(len(walls)),
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.calculate_form_factors_directivity()
    radiosity.calculate_energy_exchange(k)
    # radiosity.init_energy(source_pos)
    # histogram = radiosity.collect_energy_receiver(
    #     receiver_pos, speed_of_sound=speed_of_sound,
    #     histogram_time_resolution=time_resolution,
    #     histogram_time_length=length_histogram)

    form_factors_from_tilde = np.max(radiosity._form_factors_tilde, axis=0)
    for i in range(4):
        npt.assert_almost_equal(
            radiosity.form_factors[:4, 4:], form_factors_from_tilde[:4, 4:, i])
        npt.assert_almost_equal(
            radiosity.form_factors[:4, 4:].T, form_factors_from_tilde[4:, :4, i])

    for i in range(8):
        npt.assert_almost_equal(radiosity._form_factors_tilde[i, i, :, :], 0)
        npt.assert_almost_equal(radiosity._form_factors_tilde[:, i, i, :], 0)


def test_set_wall_scattering(sample_walls, sofa_data_diffuse):
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(
        sample_walls, 0.2)
    (data, sources, receivers) = sofa_data_diffuse
    radiosity.set_wall_scattering(np.arange(6), data, sources, receivers)
    # check shape of scattering matrix
    assert len(radiosity._scattering) == 1
    npt.assert_almost_equal(radiosity._scattering[0].shape, (4, 4, 4))
    npt.assert_array_equal(radiosity._scattering[0], 1)
    npt.assert_array_equal(radiosity._scattering_index, 0)
    # check source and receiver direction
    for i in range(6):
        assert (np.sum(
            radiosity._sources[i].cartesian*radiosity.walls_normal[i,:],
            axis=-1)>0).all()
        assert (np.sum(
            radiosity._receivers[i].cartesian*radiosity.walls_normal[i,:],
            axis=-1)>0).all()


def test_set_wall_scattering_different(sample_walls, sofa_data_diffuse):
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(
        sample_walls, 0.2)
    (data, sources, receivers) = sofa_data_diffuse
    radiosity.set_wall_scattering([0, 1, 2], data, sources, receivers)
    radiosity.set_wall_scattering([3, 4, 5], data, sources, receivers)
    # check shape of scattering matrix
    assert len(radiosity._scattering) == 2
    for i in range(2):
        npt.assert_almost_equal(radiosity._scattering[i].shape, (4, 4, 4))
        npt.assert_array_equal(radiosity._scattering[i], 1)
    npt.assert_array_equal(radiosity._scattering_index[:3], 0)
    npt.assert_array_equal(radiosity._scattering_index[3:], 1)
    # check source and receiver direction
    for i in range(6):
        assert (np.sum(
            radiosity._sources[i].cartesian*radiosity.walls_normal[i,:],
            axis=-1)>0).all()
        assert (np.sum(
            radiosity._receivers[i].cartesian*radiosity.walls_normal[i,:],
            axis=-1)>0).all()


def test_set_wall_absorption(sample_walls):
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(
        sample_walls, 0.2)
    radiosity.set_wall_absorption(
        np.arange(6), pf.FrequencyData([0.1, 0.2], [500, 1000]))
    npt.assert_array_equal(radiosity._absorption[0], [0.1, 0.2])
    npt.assert_array_equal(radiosity._absorption_index, 0)


def test_set_wall_absorption_different(sample_walls):
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(
        sample_walls, 0.2)
    radiosity.set_wall_absorption(
        [0, 1, 2], pf.FrequencyData([0.1, 0.1], [500, 1000]))
    radiosity.set_wall_absorption(
        [3, 4, 5], pf.FrequencyData([0.2, 0.2], [500, 1000]))
    npt.assert_array_equal(radiosity._absorption[0], [0.1, 0.1])
    npt.assert_array_equal(radiosity._absorption[1], [0.2, 0.2])
    npt.assert_array_equal(radiosity._absorption_index[:3], 0)
    npt.assert_array_equal(radiosity._absorption_index[3:], 1)


def test_set_air_attenuation(sample_walls):
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(
        sample_walls, 0.2)
    radiosity.set_air_attenuation(pf.FrequencyData([0.1, 0.2], [500, 1000]))
    npt.assert_array_equal(radiosity._air_attenuation, [0.1, 0.2])


def test_total_number_of_patches():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    result = sp.radiosity_fast.total_number_of_patches(points, 0.2)
    desired = 25
    assert result == desired


@pytest.mark.parametrize('patch_size', [
    1/3,
    0.5,
    1,
    ])
@pytest.mark.parametrize('k', [
    0, 1, 2, 3, 4, 5, 6,
    ])
def test_energy_exchange_simple_k1(patch_size, k,sample_walls, sofa_data_diffuse):
    # note that order k=0 means one reflection and k=1 means two reflections
    # (2nd order)
    data, sources, receivers = sofa_data_diffuse
    walls = [0, 1]

    source_pos = np.array([0.5, 0.5, 0.5])
    receiver_pos = np.array([0.5, 0.5, 0.5])
    wall_source = sample_walls[walls[0]]
    wall_receiver = sample_walls[walls[1]]
    walls = [wall_source, wall_receiver]
    length_histogram = 0.1
    time_resolution = 1e-3
    speed_of_sound = 346.18

    radiosity_old = sp.radiosity.Radiosity(
        walls, patch_size, k, length_histogram,
        speed_of_sound=speed_of_sound,
        sampling_rate=1/time_resolution, absorption=0)

    radiosity_old.run(
        sp.geometry.SoundSource(source_pos, [1, 0, 0], [0, 0, 1]))
    histogram_old = radiosity_old.energy_at_receiver(
        sp.geometry.Receiver(receiver_pos, [1, 0, 0], [0, 0, 1]), ignore_direct=True)

    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(
        walls, patch_size)

    radiosity.set_wall_scattering(
        np.arange(len(walls)), data, sources, receivers)
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.set_wall_absorption(
        np.arange(len(walls)),
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.check_visibility()
    radiosity.calculate_form_factors()
    radiosity.calculate_form_factors_directivity()
    radiosity.calculate_energy_exchange(k)
    radiosity.init_energy(source_pos)
    histogram = radiosity.collect_energy_receiver(
        receiver_pos, speed_of_sound=speed_of_sound,
        histogram_time_resolution=time_resolution,
        histogram_time_length=length_histogram)

    patches_center = []
    patches_normal = []
    patches_size = []
    for patch in radiosity_old.patch_list:
        for p in patch.patches:
            patches_center.append(p.center)
            patches_size.append(p.size)
            patches_normal.append(p.normal)
    patches_center = np.array(patches_center)
    patches_normal = np.array(patches_normal)
    patches_size = np.array(patches_size)

    npt.assert_almost_equal(
        patches_center, radiosity.patches_center)
    npt.assert_almost_equal(
        patches_normal, radiosity.patches_normal)
    npt.assert_almost_equal(
        patches_size, radiosity.patches_size)

    # compare energy exchange
    E_matrix = np.zeros((
        1,
        k+1,  # order of energy exchange
        patches_size.shape[0],  # receiver ids of patches
        radiosity_old.patch_list[0].E_n_samples,  # impulse response G_k(t)
        ))
    for k in range(radiosity.energy_exchange.shape[0]):
        for i in range(radiosity.energy_exchange.shape[1]):
            for j in range(radiosity.energy_exchange.shape[2]):
                e = radiosity.energy_exchange[k, i, j, 0]
                d = radiosity.energy_exchange[k, i, j, -1]
                samples = int(d/speed_of_sound/time_resolution)
                E_matrix[:, k, j, samples] += e
    E_matrix_old = np.concatenate(
        [p.E_matrix for p in radiosity_old.patch_list], axis=-2)
    # npt.assert_almost_equal(
    #     np.array(np.where(E_matrix>0)),
    #     np.array(np.where(E_matrix_old>0)))
    for k in range(E_matrix.shape[1]):
        npt.assert_array_equal(
            np.sum(E_matrix[:, k], axis=-1),
            np.sum(E_matrix_old[:, k], axis=-1))

    # compare histogram
    for i in range(4):
        assert np.sum(histogram[:, i])>0
        npt.assert_almost_equal(
            np.sum(histogram[:, i]), np.sum(histogram_old[0, :]))
        # npt.assert_almost_equal(histogram[:, i], histogram_old[0, :])


def test_init_source(sample_walls, sofa_data_diffuse):
    k = 1
    patch_size = 0.5
    data, sources, receivers = sofa_data_diffuse
    walls = [0, 1]

    source_pos = np.array([0.5, 0.5, 0.5])
    wall_source = sample_walls[walls[0]]
    wall_receiver = sample_walls[walls[1]]
    walls = [wall_source, wall_receiver]
    length_histogram = 0.1
    time_resolution = 1e-3
    speed_of_sound = 346.18

    radiosity_old = sp.radiosity.Radiosity(
        walls, patch_size, k, length_histogram,
        speed_of_sound=speed_of_sound,
        sampling_rate=1/time_resolution, absorption=0)
    radiosity_old.run(
        sp.geometry.SoundSource(source_pos, [1, 0, 0], [0, 0, 1]))

    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(
        walls, patch_size)

    radiosity.set_wall_scattering(
        np.arange(len(walls)), data, sources, receivers)
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.set_wall_absorption(
        np.arange(len(walls)),
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.check_visibility()
    radiosity.calculate_form_factors()
    radiosity.calculate_form_factors_directivity()
    radiosity.calculate_energy_exchange(k)
    radiosity.init_energy(source_pos)

    patches_center = []
    patches_normal = []
    patches_size = []
    for patch in radiosity_old.patch_list:
        for p in patch.patches:
            patches_center.append(p.center)
            patches_size.append(p.size)
            patches_normal.append(p.normal)
    patches_center = np.array(patches_center)
    patches_normal = np.array(patches_normal)
    patches_size = np.array(patches_size)

    energy_0, distance_0 = sp.radiosity_fast.calculate_init_energy(
        source_pos, patches_center, patches_normal,
        patches_size)

    idx =  np.where(radiosity_old.patch_list[0].E_matrix[0,0, ...]>0)
    npt.assert_almost_equal(
        energy_0[:4],
        radiosity_old.patch_list[0].E_matrix[0,0, idx[0], idx[1]]*1)
