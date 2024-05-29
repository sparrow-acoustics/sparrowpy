"""Test radiosity module."""
import numpy as np
import numpy.testing as npt
import sparapy.geometry as geo
from sparapy.radiosity_fast import DRadiosityFast
from sparapy.sound_object import SoundSource


sample_walls = [
    geo.Polygon(
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [1, 0, 0], [0, 1, 0]),
    geo.Polygon(
        [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],
        [1, 0, 0], [0, -1, 0]),
    geo.Polygon(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [1, 0, 0], [0, 0, 1]),
    geo.Polygon(
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        [1, 0, 0], [0, 0, -1]),
    geo.Polygon(
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [0, 0, 1], [1, 0, 0]),
    geo.Polygon(
        [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],
        [0, 0, 1], [-1, 0, 0]),
]

def test_init():
    radiosity = DRadiosityFast.from_polygon(sample_walls, 0.2)
    assert radiosity.speed_of_sound == 346.18
    npt.assert_almost_equal(radiosity.patches_points.shape, (150, 4, 3))
    npt.assert_almost_equal(radiosity.patches_area.shape, (150))
    npt.assert_almost_equal(radiosity.patches_center.shape, (150, 3))
    npt.assert_almost_equal(radiosity.patches_size.shape, (150, 3))
    npt.assert_almost_equal(radiosity.patches_normal.shape, (150, 3))


def test_compute_form_factors():
    radiosity = DRadiosityFast.from_polygon(sample_walls, 0.2)
    radiosity.calculate_form_factors()
    npt.assert_almost_equal(radiosity.form_factors.shape, (150, 150))


def test_init_energy():
    radiosity = DRadiosityFast.from_polygon(sample_walls, 0.2)
    (energy, distance) = radiosity.init_energy([0.5, 0.5, 0.5])
    npt.assert_array_equal(energy.shape, (150))
    npt.assert_array_equal(distance.shape, (distance))
