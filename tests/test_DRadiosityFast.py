"""Test radiosity module."""
import numpy as np
import pytest
import numpy.testing as npt
import sparapy as sp


sample_walls = [
    sp.geometry.Polygon(
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [1, 0, 0], [0, 1, 0]),
    sp.geometry.Polygon(
        [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],
        [1, 0, 0], [0, -1, 0]),
    sp.geometry.Polygon(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [1, 0, 0], [0, 0, 1]),
    sp.geometry.Polygon(
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        [1, 0, 0], [0, 0, -1]),
    sp.geometry.Polygon(
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [0, 0, 1], [1, 0, 0]),
    sp.geometry.Polygon(
        [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],
        [0, 0, 1], [-1, 0, 0]),
]

def test_init():
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(sample_walls, 0.2)
    assert radiosity.speed_of_sound == 346.18
    npt.assert_almost_equal(radiosity.patches_points.shape, (150, 4, 3))
    npt.assert_almost_equal(radiosity.patches_area.shape, (150))
    npt.assert_almost_equal(radiosity.patches_center.shape, (150, 3))
    npt.assert_almost_equal(radiosity.patches_size.shape, (150, 3))
    npt.assert_almost_equal(radiosity.patches_normal.shape, (150, 3))


def test_check_visibility():
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(sample_walls, 0.2)
    radiosity.check_visibility()
    npt.assert_almost_equal(radiosity._visibility_matrix.shape, (150, 150))
    npt.assert_array_equal(
        radiosity._visibility_matrix, radiosity._visibility_matrix.T)
    npt.assert_array_equal(radiosity._visibility_matrix[:25, :25], False)
    npt.assert_array_equal(radiosity._visibility_matrix[25:, :25], True)
    npt.assert_array_equal(radiosity._visibility_matrix[25:50, 25:50], False)
    assert np.sum(radiosity._visibility_matrix) == 25*5*25*6


def test_compute_form_factors():
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(sample_walls, 0.2)
    radiosity.check_visibility()
    radiosity.calculate_form_factors()
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
def test_calc_form_factor_perpendicular_distance(
        walls, patch_size):
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
        npt.assert_almost_equal(radiosity.form_factors[:4, 4:], patch_1.form_factors)
    else:
        npt.assert_almost_equal(radiosity.form_factors[:4, 4:], patch_1.form_factors.T)

    patch_pos = np.array([patch.center for patch in patch_2.patches])
    if (np.abs(patch_pos- radiosity.patches_center[4:, :])<1e-5).all():
        npt.assert_almost_equal(radiosity.form_factors[4:, :4], patch_2.form_factors)
    else:
        npt.assert_almost_equal(radiosity.form_factors[4:, :4], patch_2.form_factors.T)



def test_init_energy():
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(sample_walls, 0.2)
    (energy, distance) = radiosity.init_energy([0.5, 0.5, 0.5])
    npt.assert_array_equal(energy.shape, (150))
    npt.assert_array_equal(distance.shape, (150))
