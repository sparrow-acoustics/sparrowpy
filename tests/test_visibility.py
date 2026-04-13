import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest

import sparrowpy as sp

bpy = pytest.importorskip("bpy")


@pytest.mark.parametrize("origin", [np.array([0.0, 1.0, 3.0])])
@pytest.mark.parametrize("point", [np.array([0.0, 1.0, -1])])
@pytest.mark.parametrize("plpt", [np.array([1.0, 1.0, 0.0])])
@pytest.mark.parametrize("pln", [np.array([0.0, 0.0, 1.0])])
@pytest.mark.parametrize("solution", [np.array([0.0, 1.0, 0.0])])
def test_point_plane_projection(
    origin: np.ndarray,
    point: np.ndarray,
    plpt: np.ndarray,
    pln: np.ndarray,
    solution,
):
    """Ensure correct projection of rays into plane."""
    out = sp.geometry._project_to_plane(origin, point, plpt, pln)

    npt.assert_array_equal(solution, out)


@pytest.mark.parametrize(
    "point",
    [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 2.0, 0.0]),
    ],
)
@pytest.mark.parametrize(
    "plpt",
    [
        np.array(
            [
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
            ]
        ),
    ],
)
@pytest.mark.parametrize("pln", [np.array([0.0, 0.0, 1.0])])
def test_point_in_polygon(point, plpt, pln):
    """Ensure correct projection of rays into plane."""
    out = sp.geometry._point_in_polygon(
        point3d=point, polygon3d=plpt, plane_normal=pln
    )

    if abs(point[0]) > 1.0 or abs(point[1]) > 1:
        solution = False
    else:
        solution = True

    assert solution == out


@pytest.mark.parametrize(
    "point",
    [
        np.array([0.0, 0.0, 0.5]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 2.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 3.0, -1.0]),
    ],
)
@pytest.mark.parametrize("origin", [np.array([0.0, 0.0, 1.0])])
@pytest.mark.parametrize(
    "plpt",
    [
        3
        * np.array(
            [
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
            ]
        ),
        3
        * np.array(
            [
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
            ]
        ),
    ],
)
def test_basic_visibility(point, origin, plpt):
    """Test basic_visibility function."""

    pln = np.cross(plpt[1] - plpt[0], plpt[2] - plpt[1])
    pln /= np.linalg.norm(pln)

    out = sp.geometry._basic_visibility(
        eval_point=point, vis_point=origin, surf_points=plpt, surf_normal=pln
    )

    if np.dot(point - origin, pln) > 0 or point[2] >= 0:
        solution = 1
    else:
        solution = 0
    assert solution == out


def test_source_vis(basicscene):
    """Test visibility check between source and patches."""

    radi = sp.DirectionalRadiosityFast.from_polygon(
        basicscene["walls"], patch_size=1.0
    )

    radi.init_source_energy(pf.Coordinates(3.0, 3.0, 3.0))
    npt.assert_equal(
        radi._energy_init_source, np.zeros_like(radi._energy_init_source)
    )
    npt.assert_equal(
        radi._source_visibility, np.zeros_like(radi._source_visibility)
    )

    radi.init_source_energy(pf.Coordinates(0.5, 0.5, 0.5))
    npt.assert_equal(
        radi._source_visibility, np.ones_like(radi._source_visibility)
    )
    assert (radi._energy_init_source != 0).any()


def test_receiver_vis(basicscene):
    """Test visibility check between source and patches."""

    radi = sp.DirectionalRadiosityFast.from_polygon(
        basicscene["walls"], patch_size=1.0
    )

    radi.init_source_energy(pf.Coordinates(0.5, 0.5, 0.5))

    radi.calculate_energy_exchange(
        speed_of_sound=343, etc_time_resolution=0.2, etc_duration=1.0
    )

    etc = radi.collect_energy_receiver_mono(
        pf.Coordinates(
            [3.0, -5.0, 0.5],  # x
            [3.0, 5.0, 0.5],  # y
            [3.0, 4.0, 0.5],
        )
    )  # z

    npt.assert_equal(
        etc.time[0:2, 0, :],
        np.array([np.zeros((5)), np.zeros((5))]),
    )
    assert (etc.time[-1, 0, :] != 0).any()
