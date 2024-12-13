import numpy.testing as npt
import pytest
import sparapy.radiosity_fast.visibility_helpers as vh
import numpy as np
import sparapy.radiosity_fast.geometry as geom
import sparapy.radiosity_fast.blender_helpers as bh

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
    out = vh.basic_visibility(eval_point=point,vis_point=origin,
                              surf_points=plpt,surf_normal=pln)

    if (abs(point[0])/(-point[2]+1) > 1. or
        abs(point[1])/(-point[2]+1) > 1) or point[2]>0 or pln[2]<0:
        solution = 1
    else:
        solution = 0

    assert solution==out

@pytest.mark.parametrize("model", [
    "./tests/test_data/cube.blend",
    "./tests/test_data/cube_blocked.blend",
    ])
@pytest.mark.parametrize("solution", [
    np.array([0.,0.,1.]),
    np.array([0.,.5,-.5])/np.linalg.norm(np.array([0.,.5,-.5]))
    ])
def test_vis_matrix_assembly(model, solution):

    m1,m2 = bh.read_geometry_file(model)

    patches_points = np.empty((len(m1["conn"]),len(m1["conn"][0]),3))
    patches_centers = np.empty((len(m1["conn"]),3))

    for m in [m1,m2]:
        surfs=m

        surfs_points = np.empty((len(surfs["conn"]),len(surfs["conn"][0]),3))
        surfs_normals = np.empty((len(m["conn"]),3))


        for i in range(len(m1["conn"])):
            patches_points[i]=m1["verts"][m1["conn"][i]]
            patches_centers[i]=geom._calculate_center(m1["verts"][m1["conn"][i]])

        for i in range(len(m["conn"])):
            surfs_points[i]=m["verts"][m["conn"][i]]
            surfs_normals[i]=geom._calculate_center(m["verts"][m["conn"][i]])
