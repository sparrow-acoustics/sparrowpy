import numpy.testing as npt
import pytest
import sparapy.radiosity_fast.visibility_helpers as vh
import numpy as np


@pytest.mark.parametrize("origin", [np.array([0.,1.,3.])])
@pytest.mark.parametrize("point", [np.array([0.,1.,-1])])
@pytest.mark.parametrize("plpt", [np.array([1.,1.,0.])])
@pytest.mark.parametrize("pln", [np.array([0.,0.,1.])])
@pytest.mark.parametrize("solution", [np.array([0.,1.,0.])])
def test_point_plane_projection(origin: np.ndarray, point: np.ndarray,
                                plpt: np.ndarray, pln: np.ndarray, solution):
    """Ensure correct projection of rays into plane."""
    out = vh.project_to_plane(origin, point, plpt, pln)

    npt.assert_array_equal(solution,out)



@pytest.mark.parametrize("point", [
    np.array([0.,0.,0.]),
    np.array([0.,2.,0.])
    ])
@pytest.mark.parametrize("plpt", [
    np.array([[1.,1.,0.],[-1.,1.,0.],[-1.,-1.,0.],[1.,-1.,0.]])
    ])
@pytest.mark.parametrize("pln", [np.array([0.,0.,1.])])
def test_point_in_polygon(point, plpt, pln):
    """Ensure correct projection of rays into plane."""
    out = vh.point_in_polygon(point3d=point, polygon3d=plpt, plane_normal=pln)

    if abs(point[0]) > 1. or abs(point[1]) > 1:
        solution = False
    else:
        solution = True

    assert solution==out

