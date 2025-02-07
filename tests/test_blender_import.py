import numpy.testing as npt
import pytest
import numpy as np

import sparrowpy.utils.blender as bh

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
    else:
        assert len(walls["material"])==0
    assert True

@pytest.mark.parametrize("path",
                         ["./tests/test_data/cube.blend","./tests/test_data/cube.stl"])
def test_patch_generation(path):

    walls,patches = bh.read_geometry_file(path,max_patch_size=3.)

    ## check if new patches are generated
    # in case max patch size is larger than max wall side
    assert patches["conn"].shape[0]==len(walls["conn"])
