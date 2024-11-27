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


@pytest.mark.parametrize("point", [
    np.array([0.,0.,-1.]),
    np.array([0.,0.,.5]),
    np.array([0.,3.,-1.]),
    ])
@pytest.mark.parametrize("origin", [np.array([0.,0.,1.])])
@pytest.mark.parametrize("plpt", [
    np.array([[1.,1.,0.],[-1.,1.,0.],[-1.,-1.,0.],[1.,-1.,0.]])
    ])
@pytest.mark.parametrize("pln", [
    np.array([0.,0.,1.]),
    np.array([0.,.5,-.5])/np.linalg.norm(np.array([0.,.5,-.5]))
    ])
def test_basic_visibility(point, origin, plpt, pln):
    """Test basic_visibility function."""
    out = vh.basic_visibility(patch_center=point,vis_point=origin,
                              surf_points=plpt,surf_normal=pln)

    if (abs(point[0])/(-point[2]+1) > 1. or
        abs(point[1])/(-point[2]+1) > 1) or point[2]>0 or pln[2]<0:
        solution = 1
    else:
        solution = 0

    assert solution==out


@pytest.mark.parametrize("l1", [
    np.array([[-1.,0.],[1.,0.]])
    ])
@pytest.mark.parametrize("l2", [
    np.array([[0.,1.],[0.,-1.]])
    ])
@pytest.mark.parametrize("solution", [
    np.array([0.,0.])
    ])
def test_line_intersection(l1,l2,solution):
    """Test basic_visibility function."""
    out = vh.line_line_int(l1[0],l1[1],l2[0],l2[1])

    npt.assert_array_equal(out,solution)


@pytest.mark.parametrize("surf", [
    np.array([[0.,2.,0.],[-2.,2.,0.],[-2.,-2.,0.],[0.,-2.,0.]])
    ])
@pytest.mark.parametrize("patch", [
    np.array([[1.,1.,0.],[-1.,1.,0.],[-1.,-1.,0.],[1.,-1.,0.]])
    ])
@pytest.mark.parametrize("solution", [
    [[np.array([]).reshape((-1,3)),
     np.array([[0,3],[2,3]]),
     np.array([[0.,1.,0.],[0.,-1.,0.]])
    ],
    [np.array([[-1.,1.,0.],[-1.,-1.,0.]]),
     np.array([[3,0],[3,2]]),
     np.array([[0.,1.,0.],[0.,-1.,0.]])
    ]]
    ])
def test_intersection_finder(patch,surf,solution):

    for i in range(2):
        verts, conn, pts =vh.find_all_intersections(
                                                vis_point=np.array([0.,0.,1.]),
                                                patch_points=patch,
                                                patch_normal=np.array([0.,0.,1.]),
                                                surf_points=surf)

        npt.assert_array_equal(verts, solution[i][0])
        npt.assert_array_equal(conn, solution[i][1])
        npt.assert_array_equal(pts, solution[i][2])

        temp = patch
        patch = surf
        surf = temp


