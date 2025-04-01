"""Test radiosity module."""
import numpy as np
import numpy.testing as npt
import pytest
import pyfar as pf

import sparrowpy as sp


create_reference_files = False

def test_init_from_polygon(sample_walls):
    radiosity = sp.DRadiosityFast.from_polygon(sample_walls, 0.2)
    npt.assert_almost_equal(radiosity.patches_points.shape, (150, 4, 3))
    npt.assert_almost_equal(radiosity.patches_area.shape, (150))
    npt.assert_almost_equal(radiosity.patches_center.shape, (150, 3))
    npt.assert_almost_equal(radiosity.patches_size.shape, (150, 3))
    npt.assert_almost_equal(radiosity.patches_normal.shape, (150, 3))

@pytest.mark.parametrize('filename', [
    "tests/test_data/cube.blend",
    "tests/test_data/cube.stl",
    ])
def test_init_from_file(filename):
    radiosity = sp.DRadiosityFast.from_file(filename,np.sqrt(2))
    npt.assert_almost_equal(radiosity.patches_points.shape, (12, 3, 3))
    npt.assert_almost_equal(radiosity.patches_area.shape, (12))
    npt.assert_almost_equal(radiosity.patches_center.shape, (12, 3))
    npt.assert_almost_equal(radiosity.patches_size.shape, (12, 3))
    npt.assert_almost_equal(radiosity.patches_normal.shape, (12, 3))


@pytest.mark.parametrize('filename', [
    "tests/test_data/sample_walls.blend",
    ])
def test_init_comparison(filename, sample_walls):
    radifile = sp.DRadiosityFast.from_file(filename)
    radipoly = sp.DRadiosityFast.from_polygon(sample_walls, patch_size=1)
    npt.assert_equal(radifile.patches_points.shape,
                     radipoly.patches_points.shape)
    npt.assert_equal(radifile.patches_area.shape,
                     radipoly.patches_area.shape)
    npt.assert_equal(radifile.patches_center.shape,
                     radipoly.patches_center.shape)
    npt.assert_equal(radifile.patches_size.shape,
                     radipoly.patches_size.shape)
    npt.assert_equal(radifile.patches_normal.shape,
                     radipoly.patches_normal.shape)

    radifile = sp.DRadiosityFast.from_file(filename, patch_size=.5,
                                           auto_patches=False)
    radipoly = sp.DRadiosityFast.from_polygon(sample_walls, patch_size=.5)
    npt.assert_equal(radifile.patches_points.shape,
                     radipoly.patches_points.shape)
    npt.assert_equal(radifile.patches_area.shape,
                     radipoly.patches_area.shape)
    npt.assert_equal(radifile.patches_center.shape,
                     radipoly.patches_center.shape)
    npt.assert_equal(radifile.patches_size.shape,
                     radipoly.patches_size.shape)
    npt.assert_equal(radifile.patches_normal.shape,
                     radipoly.patches_normal.shape)


def test_check_visibility(sample_walls):
    radiosity = sp.DirectionalRadiosityFast.from_polygon(sample_walls, 0.2)
    radiosity.bake_geometry()
    npt.assert_almost_equal(radiosity._visibility_matrix.shape, (150, 150))
    # npt.assert_array_equal(
    #     radiosity._visibility_matrix, radiosity._visibility_matrix.T)
    npt.assert_array_equal(radiosity._visibility_matrix[:25, :25], False)
    npt.assert_array_equal(radiosity._visibility_matrix[:25, 25:], True)
    npt.assert_array_equal(radiosity._visibility_matrix[25:50, 25:50], False)
    assert np.sum(radiosity._visibility_matrix) == 25*5*25*6/2


def test_check_visibility_wrapper(sample_walls):
    radiosity = sp.DirectionalRadiosityFast.from_polygon(sample_walls, 0.2)
    radiosity.bake_geometry()
    visibility_matrix = sp.radiosity_fast.geometry.check_visibility(
        radiosity.patches_center, radiosity.patches_normal,
        radiosity.patches_points)
    npt.assert_almost_equal(radiosity._visibility_matrix, visibility_matrix)


def test_compute_form_factors(sample_walls):
    radiosity = sp.DirectionalRadiosityFast.from_polygon(sample_walls, 0.2)
    radiosity.bake_geometry()
    npt.assert_almost_equal(radiosity.form_factors.shape, (150, 150))
    radiosity.bake_geometry(ff_method='universal')
    npt.assert_almost_equal(radiosity.form_factors.shape, (150, 150))



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

    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        walls, patch_size)
    data, sources, receivers = sofa_data_diffuse
    radiosity.set_wall_scattering(
        np.arange(len(walls)), data, sources, receivers)
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.set_wall_absorption(
        np.arange(len(walls)),
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.bake_geometry()

    form_factors_from_tilde = np.max(radiosity._form_factors_tilde, axis=2)
    for i in range(4):
        npt.assert_almost_equal(
            radiosity.form_factors[:4, 4:], form_factors_from_tilde[:4, 4:, i])
        npt.assert_almost_equal(
            radiosity.form_factors[:4, 4:].T,
            form_factors_from_tilde[4:, :4, i])

    for i in range(8):
        npt.assert_almost_equal(radiosity._form_factors_tilde[i, i, :, :], 0)


def test_set_wall_scattering(sample_walls, sofa_data_diffuse):
    radiosity = sp.DirectionalRadiosityFast.from_polygon(
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
    radiosity = sp.DirectionalRadiosityFast.from_polygon(
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
    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        sample_walls, 0.2)
    radiosity.set_wall_absorption(
        np.arange(6), pf.FrequencyData([0.1, 0.2], [500, 1000]))
    npt.assert_array_equal(radiosity._absorption[0], [0.1, 0.2])
    npt.assert_array_equal(radiosity._absorption_index, 0)


def test_set_wall_absorption_different(sample_walls):
    radiosity = sp.DirectionalRadiosityFast.from_polygon(
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
    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        sample_walls, 0.2)
    radiosity.set_air_attenuation(pf.FrequencyData([0.1, 0.2], [500, 1000]))
    npt.assert_array_equal(radiosity._air_attenuation, [0.1, 0.2])


def test_total_number_of_patches():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    result = sp.radiosity_fast.geometry.total_number_of_patches(points, 0.2)
    desired = 25
    assert result == desired
