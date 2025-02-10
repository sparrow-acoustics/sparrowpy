import numpy.testing as npt
import pytest
import numpy as np
import matplotlib.pyplot as plt
import sparrowpy.utils.blender as bh
import os

@pytest.mark.parametrize("path",
                         ["./tests/test_data/cube.blend","./tests/test_data/cube.stl"])

def test_basic_features(path):

    walls,patches = bh.read_geometry_file(path)

    assert len(walls["conn"])==12
    assert len(walls["normal"])==len(walls["conn"])
    assert len(walls["verts"])==8
    assert patches["conn"].shape[1]==3
    assert patches["verts"].shape[1]==3
    assert len(walls["normal"][0])==3
    assert np.max(patches["wall_ID"])==len(walls["conn"])-1

@pytest.mark.parametrize("path",
                         ["./tests/test_data/cube.blend","./tests/test_data/cube.stl"])
def test_material_assignment(path):
    walls,_ = bh.read_geometry_file(path)
    if path.endswith(".blend"):
        assert len(walls["material"])==len(walls["conn"])
        assert isinstance(walls["material"][0], str)
        for i in range(len(walls["conn"])):
            if i == 2:
                assert walls["material"][i]=="matA"
            else:
                assert walls["material"][i]=="matB"
    else:
        assert len(walls["material"])==0
    assert True

@pytest.mark.parametrize("path",
                         ["./tests/test_data/cube.blend","./tests/test_data/cube.stl",
                          "./tests/test_data/disk_10sides.blend"])
def test_patch_generation(path):

    model_name = os.path.split(path)[1].replace(".","_")

    walls,patches = bh.read_geometry_file(path,max_patch_size=3.)

    ## check if new patches are generated
    # in case max patch size is larger than max wall side
    npt.assert_equal(patches["conn"],np.array(walls["conn"]))
    npt.assert_equal(patches["verts"],walls["verts"])

    side = np.linalg.norm(walls["verts"][1]-walls["verts"][0])

    ## check if n patches follows the max_patch_size change
    walls,p0 = bh.read_geometry_file(path,max_patch_size=.5*side)
    _,p1 = bh.read_geometry_file(path,max_patch_size=.25*side)

    plot_mesh_data(mesh=walls, model_name=model_name, refinement="_walls")

    assert p1["conn"].shape[0]>p0["conn"].shape[0]

    level = ["_rough","_fine"]

    for i,plist in enumerate([p0,p1]):
        plot_mesh_data(mesh=plist, model_name=model_name,refinement=level[i])

@pytest.mark.parametrize("simplepath",
                         ["./tests/test_data/cube_simple.blend"])
@pytest.mark.parametrize("path",
                         ["./tests/test_data/cube.blend"])
def test_patch_generation_options(simplepath,path):
    squarewalls,squarepatches = bh.read_geometry_file(simplepath,
                                                      max_patch_size=.2,
                                                      auto_walls=False,
                                                      auto_patches=False)
    
    # check that user-defined walls are same as user-defined patches
    npt.assert_equal(np.array(squarewalls["conn"]), squarepatches["conn"])
    npt.assert_equal(np.array(squarewalls["verts"]), squarepatches["verts"])
    
    triwalls,tripatches = bh.read_geometry_file(simplepath,
                                                      max_patch_size=.2,
                                                      auto_walls=True,
                                                      auto_patches=False)
    
    # check that auto walls are same as non-auto patches
    npt.assert_equal(np.array(triwalls["conn"]), tripatches["conn"])
    npt.assert_equal(np.array(triwalls["verts"]), tripatches["verts"])
    
    # check that vertices are the same regardless of shape as long
    # as patches are not generated automatically
    npt.assert_equal(np.array(triwalls["verts"]), squarepatches["verts"])
    
    # check auto walls are same as user walls (no polygons are to be merged in simple cube)
    assert len(triwalls["conn"]) == len(squarewalls["conn"])
    
    
    with pytest.warns() as rec:
        simplewalls,simplepatches = bh.read_geometry_file(simplepath,
                                                      max_patch_size=.2,
                                                      auto_walls=False,
                                                      auto_patches=True)
        walls, patches = bh.read_geometry_file( path,
                                                max_patch_size=.2,
                                                auto_walls=False,
                                                auto_patches=False )
        
    assert len(simplewalls["conn"]) < simplepatches["conn"].shape[0]
    npt.assert_equal(np.array(simplewalls["verts"]), -np.array(walls["verts"])) # not sure why cube is rotated
    
    # check that auto_patches and irregular wall shape both force triangulation of the walls
    assert len(walls["conn"]) == len(simplewalls["conn"])
    # check wall triangulation
    assert len(walls["conn"]) == 2*len(squarewalls["conn"])
    
    # check that warning is thrown if user-defined walls are of inconsistent shapes
    assert len(rec)==2
    assert issubclass(rec[1].category, UserWarning)
    assert str(rec[1].message) == "User-input patches have inconsistent shapes.\nA rough triangulation will be applied."
    assert issubclass(rec[0].category, RuntimeWarning)
    assert str(rec[0].message) == "A rough triangulation pass may be applied to user-defined walls for auto patch generation."

    

def plot_mesh_data(mesh,model_name, refinement):
    
    fig = plt.figure()
    if "disk" not in model_name:
        ax = fig.add_subplot(111, projection='3d')
        for ids in mesh["conn"]:
            ids=np.append(ids,ids[0])
            ax.plot(mesh["verts"][ids,0],mesh["verts"][ids,1],mesh["verts"][ids,2],"b-")
    else:
        ax = fig.add_subplot()
        for ids in mesh["conn"]:
            ids=np.append(ids,ids[0])
            ax.plot(mesh["verts"][ids,0],mesh["verts"][ids,1],"b-")
        ax.axis("equal")
            
    plt.savefig("tests/test_data/patch_gen_"+model_name+refinement+".png")