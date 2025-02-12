import numpy.testing as npt
import pytest
import numpy as np
import matplotlib.pyplot as plt
import sparrowpy.utils.blender as bh
import os

@pytest.mark.parametrize("path",
                         ["./tests/test_data/cube_simple.blend","./tests/test_data/cube.stl"])

def test_geometry_loading(path):

    walls = bh.read_geometry_file(path, auto_walls=True)

    assert walls["conn"].shape[0]==6
    assert walls["normal"].shape[0]==walls["conn"].shape[0]
    assert walls["verts"].shape[0]==8
    assert walls["normal"].shape[1]==3
    
def test_auto_wall():
    """Test if auto/manual wall assignment is working as expected"""
    # check if sub-walls of consistent material are merged or individually picked up
    simple_merge = bh.read_geometry_file("./tests/test_data/cube_simple.blend", auto_walls=True)
    simple_split = bh.read_geometry_file("./tests/test_data/cube_simple.blend", auto_walls=False)
    
    assert simple_merge["conn"].shape[0]!=simple_split["conn"].shape[0]
    assert simple_split["conn"].shape[0]==24
    
    # check if sub-walls of different material are merged or individually picked up
    dual_merge = bh.read_geometry_file("./tests/test_data/cube_subdiv.blend", auto_walls=True)
    dual_split = bh.read_geometry_file("./tests/test_data/cube_subdiv.blend", auto_walls=False)
    
    assert dual_merge["conn"].shape[0]==10
    npt.assert_array_equal(dual_merge["conn"],dual_split["conn"])


@pytest.mark.parametrize("path",
                         ["./tests/test_data/cube_simple.blend","./tests/test_data/cube.stl"])
def test_material_assignment(path):
    """Check that material is correctly assigned from model."""
    walls = bh.read_geometry_file(path)
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
