"""Test radiosity module."""
import numpy as np
import numpy.testing as npt
import sparrowpy.geometry as geo
import sparrowpy as sp
from sparrowpy.sound_object import SoundSource


def test_patches():
    """Test Patches class."""
    poly = geo.Polygon(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [0, 1, 0], [0, 0, 1])
    patches = sp.PatchesKang(poly, 0.1, [], 0)
    assert len(patches.patches) == 100
    for patch in patches.patches:
        # check if area is always same
        dimensions = np.max(patch.pts, axis=0) - np.min(patch.pts, axis=0)
        dimensions = dimensions[dimensions > 0]
        area = dimensions[0]*dimensions[1]
        npt.assert_almost_equal(area, 0.01)
    npt.assert_almost_equal(patches.patches[0].pts, poly.pts/10)


def test_patches_dim2():
    """Test Patches class with 2D polygon."""
    poly = geo.Polygon(
        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], [0, 1, 0], [1, 0, 0])
    patches = sp.PatchesKang(poly, 0.1, [], 0)
    assert len(patches.patches) == 100
    for patch in patches.patches:
        # check if area is always same
        dimensions = np.max(patch.pts, axis=0) - np.min(patch.pts, axis=0)
        dimensions = dimensions[dimensions > 0]
        area = dimensions[0]*dimensions[1]
        npt.assert_almost_equal(area, 0.01)
    npt.assert_almost_equal(patches.patches[0].pts, poly.pts/10)


def test_form_factor():
    """Test form_factor function."""
    poly1 = geo.Polygon(
        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], [0, 1, 0], [1, 0, 0])
    patches1 = sp.PatchesKang(poly1, 0.2, [1], 0)
    poly2 = geo.Polygon(
        [[2, 0, 0], [2, 1, 0], [2, 1, 1], [2, 0, 1]], [2, 1, 0], [1, 0, 0])
    patches2 = sp.PatchesKang(poly2, 0.2, [0], 1)
    patches1.calculate_form_factor([patches1, patches2])


def test_calculate_form_factors():
    """Test calculate_form_factors function."""
    # set geometry
    X = 8
    Y = 4
    Z = 4
    patch_size = 2

    ground = geo.Polygon(
        [[0, 0, 0], [X, 0, 0], [X, Y, 0], [0, Y, 0]], [1, 0, 0], [0, 0, 1])
    A_wall = geo.Polygon(
        [[0, 0, 0], [X, 0, 0], [X, 0, Z], [0, 0, Z]], [1, 0, 0], [0, 1, 0])
    B_wall = geo.Polygon(
        [[0, Y, 0], [X, Y, 0], [X, Y, Z], [0, Y, Z]], [1, 0, 0], [0, -1, 0])

    ground_patches = sp.PatchesKang(ground, patch_size, [1, 2], 0)
    A_wall_patches = sp.PatchesKang(A_wall, patch_size, [0, 2], 1)
    B_wall_patches = sp.PatchesKang(B_wall, patch_size, [0, 1], 2)

    patches_list = [ground_patches, A_wall_patches, B_wall_patches]

    # calculate form factor
    A_wall_patches.calculate_form_factor(patches_list)


def test_init_energy_matrix():
    """Test init_energy_matrix function."""
    X = 8
    Y = 4
    patch_size = 2
    ground = geo.Polygon(
        [[0, 0, 0], [X, 0, 0], [X, Y, 0], [0, Y, 0]], [1, 0, 0], [0, 0, 1])
    ground_patches = sp.PatchesKang(ground, patch_size, [1, 2], 0)
    max_order_k = 2
    ir_length_s = 1
    sampling_rate = 1000
    source = SoundSource([2, 2, 1], [0, 1, 0], [0, 0, 1])
    ground_patches.init_energy_exchange(
        max_order_k, ir_length_s, source, sampling_rate=sampling_rate,
        speed_of_sound=343)
    assert ground_patches.E_n_samples == 1000
    assert ground_patches.E_sampling_rate == sampling_rate
    npt.assert_array_equal(
        ground_patches.E_matrix.shape, (1, max_order_k+1, 8, 1000))
