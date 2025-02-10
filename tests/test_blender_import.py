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
    _,p0 = bh.read_geometry_file(path,max_patch_size=.5*side)
    _,p1 = bh.read_geometry_file(path,max_patch_size=.25*side)

    assert p1["conn"].shape[0]>p0["conn"].shape[0]

    level = ["_rough","_fine"]

    for i,plist in enumerate([p0,p1]):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for ids in plist["conn"]:
            ids=np.append(ids,ids[0])
            ax.plot(plist["verts"][ids,0],plist["verts"][ids,1],plist["verts"][ids,2],"b-")

        plt.savefig("tests/test_data/patch_gen_"+model_name+level[i]+".png")

