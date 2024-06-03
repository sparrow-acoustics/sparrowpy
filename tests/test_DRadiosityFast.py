"""Test radiosity module."""
import numpy as np
import numpy.testing as npt
import pytest
import pyfar as pf

import sparapy as sp

def test_init(sample_walls):
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(sample_walls, 0.2)
    assert radiosity.speed_of_sound == 346.18
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
    ])
def test_calc_form_factor_dir_perpendicular_distance(
        sample_walls, walls, patch_size, sofa_data_diffuse):
    """Test form factor calculation for perpendicular walls."""
    source_pos = np.array([0.5, 0.5, 0.5])
    receiver_pos = np.array([0.5, 0.5, 0.5])
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
    data, sources, receivers = sofa_data_diffuse
    radiosity.set_wall_scattering(
        np.arange(2), data, sources, receivers)
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.set_wall_absorption(
        np.arange(2),
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.calculate_form_factors_directivity()
    radiosity.calculate_energy_exchange(3)
    radiosity.init_energy(source_pos)
    histogram = radiosity.collect_energy_receiver(receiver_pos, histogram_time_resolution=1e-3, histogram_time_length=1)


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


def test_init_energy(sample_walls):
    source_pos = np.array([0.5, 0.5, 0.5])
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(
        sample_walls, 0.2)
    (energy, distance) = radiosity.init_energy(source_pos)
    npt.assert_array_equal(energy.shape, (150))
    npt.assert_array_equal(distance.shape, (150))


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


def test_init_energy_wrapper(sample_walls):
    source_pos = np.array([0.5, 0.5, 0.5])
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(
        sample_walls, 0.2)
    (energy, distance) = radiosity.init_energy(source_pos)
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(
        sample_walls, 0.2)
    (energy_2, distance_2) = sp.radiosity_fast.calculate_init_energy(
            source_pos,
            radiosity.patches_center, radiosity.patches_normal,
            radiosity.patches_size)
    npt.assert_array_equal(energy.shape, energy_2.shape)
    npt.assert_array_equal(distance.shape, distance_2.shape)
    npt.assert_array_equal(energy, energy_2)
    npt.assert_array_equal(distance, distance_2)
