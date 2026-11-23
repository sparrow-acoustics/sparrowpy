"""Regression tests for points level with a polygon vertex."""
import numpy as np
import numpy.testing as npt
import pytest
import sparrowpy as sp

# triangle in the plane y=3; its vertex (6, 3, 2) is the rightmost point
TRIANGLE = np.array([[4.5, 3., 1.625], [5.047, 3., 2.625], [6., 3., 2.]])
NORMAL = np.array([0., -1., 0.])


@pytest.mark.parametrize(("point", "inside"), [
    # level with the rightmost vertex, left of the triangle
    (np.array([0.5, 3., 2.]), False),
    # level with the rightmost vertex, inside the triangle
    (np.array([5.5, 3., 2.]), True),
    # level with the lower vertex (a crossing, not a touch)
    (np.array([5., 3., 1.625]), False),
    (np.array([4., 3., 1.625]), False),
    # clearly inside and outside
    (np.array([5.2, 3., 2.1]), True),
    (np.array([7., 3., 2.]), False),
])
def test_point_level_with_vertex(point, inside):
    assert sp.geometry._point_in_polygon(
        point3d=point, polygon3d=TRIANGLE, plane_normal=NORMAL) == inside


@pytest.mark.parametrize(("point", "inside"), [
    # left of the square, level with its top and bottom side
    (np.array([-3., 1., 0.]), False),
    (np.array([-3., -1., 0.]), False),
    # inside, and left of the square level with its center
    (np.array([0., 0., 0.]), True),
    (np.array([-3., 0., 0.]), False),
    # on the boundary counts as inside: on each side and on a vertex
    (np.array([1., 0., 0.]), True),
    (np.array([-1., 0., 0.]), True),
    (np.array([0., 1., 0.]), True),
    (np.array([0., -1., 0.]), True),
    (np.array([1., 1., 0.]), True),
])
def test_point_level_with_horizontal_side(point, inside):
    square = np.array([[1., 1., 0.], [-1., 1., 0.], [-1., -1., 0.],
                       [1., -1., 0.]])
    assert sp.geometry._point_in_polygon(
        point3d=point, polygon3d=square,
        plane_normal=np.array([0., 0., 1.])) == inside


def test_visibility_not_blocked_by_polygon_level_with_vertex():
    """The segment between both patch centers crosses the blocker's plane
    level with its rightmost vertex, but left of it.
    """
    patches_points = np.array([
        [[0., 4/3, 2.125], [1., 4/3, 2.875], [0.5, 4/3, 2.875]],
        [[0.5, 5., 0.75], [0., 5., 1.5], [1., 5., 1.5]],
        TRIANGLE,
    ])
    patches_normal = np.array([[0., 1., 0.], [0., -1., 0.], NORMAL])
    visibility = sp.geometry._check_patch2patch_visibility(
        patches_points.mean(axis=1), patches_normal, patches_points)
    npt.assert_equal(visibility[0, 1], True)
