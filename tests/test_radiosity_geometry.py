"""Contains tests for the geometry module.

Geometry and wall directivity functionality in the radiosity package.
"""
import numpy as np
import numpy.testing as npt
import sparrowpy.geometry as geo
from sparrowpy.sound_object import Receiver, SoundSource


def test_polygon_defaults():
    """Test Polygon class."""
    poly = geo.Polygon(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [0, 1, 0], [0, 0, 1])
    npt.assert_equal(poly.pts[0], np.array([0, 0, 0]))
    npt.assert_equal(poly.pts[1], np.array([1, 0, 0]))
    npt.assert_equal(poly.pts[2], np.array([1, 1, 0]))
    npt.assert_equal(poly.pts[3], np.array([0, 1, 0]))
    npt.assert_almost_equal(poly.up_vector, np.array([0, 1, 0]))
    npt.assert_allclose(poly.normal, np.array([0, 0, 1]))


def test_polygon_center():
    """Test center property of Polygon."""
    poly = geo.Polygon(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [0, 1, 0], [0, 0, 1])
    npt.assert_allclose(poly.center, np.array([0.5, .5, 0]))

def test_polygon_size():
    """Test size property of Polygon."""
    poly = geo.Polygon(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [0, 1, 0], [0, 0, 1])
    npt.assert_allclose(poly.size, np.array([1, 1, 0]))


def test_polygon_surface_normal():
    """Test surface_normal property of Polygon."""
    poly = geo.Polygon(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [0, 1, 0], [0, 0, 1])
    npt.assert_equal(poly.normal, np.array([0, 0, 1]))
    npt.assert_equal(poly.up_vector, np.array([0, 1, 0]))


def test_polygon_n_points():
    """Test n_points property of Polygon."""
    poly = geo.Polygon(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [0, 1, 0], [0, 0, 1])
    assert poly.n_points == 4


def test_polygon_intersection():
    """Test intersection method of Polygon."""
    poly = geo.Polygon(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [0, 1, 0], [0, 0, 1])
    inter = poly.intersection([0.5, 0.5, 1], [0, 0, -1])
    npt.assert_allclose(inter, np.array([0.5, 0.5, 0]))
    npt.assert_allclose(poly.normal, np.array([0, 0, 1]))


def test_polygon_no_intersection():
    """Test intersection method of Polygon."""
    poly = geo.Polygon(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [0, 1, 0], [0, 0, 1])
    inter = poly.intersection([0.5, 0.5, 1], [0, 0, 1])
    assert inter is None
    npt.assert_allclose(poly.normal, np.array([0, 0, 1]))


def test_environment():
    """Test Environment class."""
    speed_of_sound = 346.2
    polygon = geo.Polygon(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [0, 1, 0], [0, 0, 1])
    source = SoundSource([0, 0, 0], [1, 0, 0], [0, 0, 1])
    receiver = Receiver([0, 0, 0], [1, 0, 0], [0, 0, 1])
    # test all parameter
    env = geo._Environment([polygon], source, receiver, speed_of_sound)
    assert env.polygons == [polygon]
    assert env.speed_of_sound == speed_of_sound
    assert env.receiver == receiver
    assert env.source == source


def test_environment_defaults():
    """Test Environment class."""
    polygon = geo.Polygon(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [0, 1, 0], [0, 0, 1])
    source = SoundSource([0, 0, 0], [1, 0, 0], [0, 0, 1])
    receiver = Receiver([0, 0, 0], [1, 0, 0], [0, 0, 1])
    # test all parameter
    env = geo._Environment([polygon], source, receiver, speed_of_sound=346.18)
    assert env.polygons == [polygon]
    assert env.speed_of_sound == 346.18
    assert env.receiver == receiver
    assert env.source == source


def test_polygon_to_dict():
    """Test to_dict method of Polygon."""
    # Arrange
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    up_vector = np.array([0, 1, 0])
    normal = np.array([0, 0, 1])
    polygon = geo.Polygon(points, up_vector, normal)

    # Act
    result = polygon.to_dict()

    # Assert
    assert isinstance(result, dict)
    assert 'pts' in result
    assert 'up_vector' in result
    assert 'normal' in result
    np.testing.assert_array_equal(result['pts'], points)
    np.testing.assert_array_equal(result['up_vector'], up_vector)
    np.testing.assert_array_equal(result['normal'], normal)


def test_polygon_from_dict():
    """Test from_dict method of Polygon."""
    # Arrange
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    up_vector = np.array([0, 1, 0])
    normal = np.array([0, 0, 1])
    polygon_dict = {
        'pts': points.tolist(),
        'up_vector': up_vector.tolist(),
        'normal': normal.tolist(),
    }

    # Act
    polygon = geo.Polygon.from_dict(polygon_dict)

    # Assert
    assert isinstance(polygon, geo.Polygon)
    np.testing.assert_array_equal(polygon.pts, points)
    np.testing.assert_array_equal(polygon.up_vector, up_vector)
    np.testing.assert_array_equal(polygon._normal, normal)
