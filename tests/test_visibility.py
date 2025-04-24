import numpy.testing as npt
import pytest
import numpy as np
import pyfar as pf
import sparrowpy as sp
bpy = pytest.importorskip("bpy")
import sparrowpy.utils.blender as bh  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

@pytest.mark.parametrize("origin", [np.array([0.,1.,3.])])
@pytest.mark.parametrize("point", [np.array([0.,1.,-1])])
@pytest.mark.parametrize("plpt", [np.array([1.,1.,0.])])
@pytest.mark.parametrize("pln", [np.array([0.,0.,1.])])
@pytest.mark.parametrize("solution", [np.array([0.,1.,0.])])
def test_point_plane_projection(origin: np.ndarray, point: np.ndarray,
                                plpt: np.ndarray, pln: np.ndarray, solution):
    """Ensure correct projection of rays into plane."""
    out = sp.geometry._project_to_plane(origin, point, plpt, pln)

    npt.assert_array_equal(solution,out)



@pytest.mark.parametrize("point", [
    np.array([0.,0.,0.]),
    np.array([0.,2.,0.]),
    ])
@pytest.mark.parametrize("plpt", [
    np.array([[1.,1.,0.],[-1.,1.,0.],[-1.,-1.,0.],[1.,-1.,0.]]),
    ])
@pytest.mark.parametrize("pln", [np.array([0.,0.,1.])])
def test_point_in_polygon(point, plpt, pln):
    """Ensure correct projection of rays into plane."""
    out = sp.geometry._point_in_polygon(point3d=point,
                                 polygon3d=plpt,
                                 plane_normal=pln)

    if abs(point[0]) > 1. or abs(point[1]) > 1:
        solution = False
    else:
        solution = True

    assert solution==out


@pytest.mark.parametrize("point", [
    np.array([0.,0.,.5]),
    np.array([0.,0.,0.]),
    np.array([0.,0.,2.]),
    np.array([0.,0.,-1.]),
    np.array([0.,3.,-1.]),
    ])
@pytest.mark.parametrize("origin", [np.array([0.,0.,1.])])
@pytest.mark.parametrize("plpt", [
    3*np.array([[1.,1.,0.],[-1.,1.,0.],[-1.,-1.,0.],[1.,-1.,0.]]),
    3*np.array([[1.,1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.],[1.,-1.,1.]]),
    ])

def test_basic_visibility(point, origin, plpt):
    """Test basic_visibility function."""

    pln = np.cross(plpt[1]-plpt[0],plpt[2]-plpt[1])
    pln /= np.linalg.norm(pln)

    out = sp.geometry._basic_visibility(eval_point=point,vis_point=origin,
                              surf_points=plpt,surf_normal=pln)

    if (np.dot(point-origin,pln)>0 or
        point[2]>=0 ):
        solution = 1
    else:
        solution = 0
    assert solution==out

@pytest.mark.parametrize("model", [
    "./tests/test_data/cube_simple.blend",
    "./tests/test_data/cube.blend",
    "./tests/test_data/cube_blocked.blend",
    ])
def test_vis_matrix_assembly(model):
    """Check if visibility matrices are correctly assembled."""
    m1,m2 = bh.read_geometry_file(model)

    patches_points = np.empty((len(m1["conn"]),len(m1["conn"][0]),3))
    patches_centers = np.empty((len(m1["conn"]),3))
    patches_normals = np.empty_like(patches_centers)

    for i in range(len(m1["conn"])):
            patches_points[i]=m1["verts"][m1["conn"][i]]
            patches_centers[i]=sp.geometry._calculate_center(m1["verts"][m1["conn"][i]])
            patches_normals[i]=np.cross(m1["verts"][m1["conn"][i]][1]
                                            -m1["verts"][m1["conn"][i]][0],
                                      m1["verts"][m1["conn"][i]][2]
                                            -m1["verts"][m1["conn"][i]][0])
            patches_normals[i]/=np.linalg.norm(patches_normals[i])

    for m in [m1,m2]:
        surfs=m

        surfs_points = []
        surfs_normals =list([np.empty((3,))]*len(surfs["conn"]))

        if model=="./tests/test_data/cube.blend":
            solution=np.zeros((patches_centers.shape[0],patches_centers.shape[0]),
                           dtype=bool)
            for i in range(solution.shape[0]):
                for j in range(i+1,solution.shape[1]):
                    ray=patches_centers[j]-patches_centers[i]
                    ray/=np.linalg.norm(ray)
                    if np.dot(patches_normals[i], ray)>1e-6:
                        solution[i,j]=True
        elif model=="./tests/test_data/cube_blocked.blend":
            solution=np.zeros((patches_centers.shape[0],patches_centers.shape[0]),
                            dtype=bool)
            for ids in A:
                solution[ids[0],ids[1]]=True
        elif model=="./tests/test_data/cube_simple.blend":
            solution=np.zeros((patches_centers.shape[0],patches_centers.shape[0]),
                            dtype=bool)
            for i in range(solution.shape[0]):
                for j in range(i+1,solution.shape[1]):
                        solution[i,j]=True

        for i in range(len(surfs["conn"])):
            surfs_points.append(surfs["verts"][surfs["conn"][i]])
            surfs_normals[i]=np.cross(surfs["verts"][m["conn"][i]][1]
                                            -m["verts"][m["conn"][i]][0],
                                      m["verts"][m["conn"][i]][2]
                                            -m["verts"][m["conn"][i]][0])
            surfs_normals[i]/=np.linalg.norm(surfs_normals[i])

        vis_matrix = sp.geometry._check_patch2patch_visibility(
                                            patches_center=patches_centers,
                                           surf_normal=surfs_normals,
                                           surf_points=surfs_points)

        plt.imsave(model[:-6]+"_vis.pdf",vis_matrix)
        plt.imsave(model[:-6]+"_sol.pdf",solution)
        npt.assert_array_equal(vis_matrix,solution)

A=[
    [0,1],
    [0,3],
    [0,4],
    [0,5],
    [0,6],
    [0,9],
    [0,14],
    [1,3],
    [1,4],
    [1,6],
    [1,7],
    [1,9],
    [1,14],
    [2,8],
    [2,9],
    [2,10],
    [2,11],
    [2,12],
    [2,13],
    [2,15],
    [3,5],
    [3,6],
    [3,7],
    [3,8],
    [3,9],
    [3,10],
    [3,11],
    [3,12],
    [3,13],
    [3,14],
    [3,15],
    [4,5],
    [4,6],
    [4,7],
    [4,9],
    [4,14],
    [5,6],
    [5,7],
    [5,9],
    [5,14],
    [6,7],
    [6,9],
    [6,14],
    [7,9],
    [7,14],
    [8,9],
    [8,10],
    [8,11],
    [8,12],
    [8,15],
    [9,10],
    [9,12],
    [9,13],
    [9,15],
    [10,11],
    [10,12],
    [10,13],
    [10,15],
    [11,12],
    [11,13],
    [11,15],
    [12,13],
    [13,15],
    [11,12],
   ]


def test_source_vis(basicscene):
    """Test visibility check between source and patches."""

    radi=sp.DirectionalRadiosityFast.from_polygon(basicscene["walls"],
                                                  patch_size=1.)

    radi.init_source_energy(pf.Coordinates(3.,3.,3.))
    npt.assert_equal(radi._energy_init_source,
                     np.zeros_like(radi._energy_init_source))
    npt.assert_equal(radi._source_visibility,
                     np.zeros_like(radi._source_visibility))

    radi.init_source_energy(pf.Coordinates(.5, .5, .5))
    npt.assert_equal(radi._source_visibility,
                     np.ones_like(radi._source_visibility))
    assert (radi._energy_init_source != 0).any()


def test_receiver_vis(basicscene):
    """Test visibility check between source and patches."""

    radi=sp.DirectionalRadiosityFast.from_polygon(basicscene["walls"],
                                                  patch_size=1.)

    radi.init_source_energy(pf.Coordinates(.5,.5,.5))

    radi.calculate_energy_exchange(speed_of_sound=343,
                                   etc_time_resolution=.2,
                                   etc_duration=1.)

    etc = radi.collect_energy_receiver_mono(pf.Coordinates([3.,-5.,.5],  #x
                                                           [3., 5.,.5],  #y
                                                           [3., 4.,.5])) #z

    npt.assert_equal(etc.time[0:2,0,:],
                     np.array([np.zeros((5)),
                              np.zeros((5))]),
                     )
    assert (etc.time[-1,0,:] != 0).any()
