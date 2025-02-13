import numpy.testing as npt
import pytest
import numpy as np
import matplotlib.pyplot as plt
import sparrowpy.utils.blender as bh
import os
import bpy
import bmesh
from sparrowpy.radiosity_fast.visibility_helpers import point_in_polygon

@pytest.mark.parametrize("path",
                         ["./tests/test_data/cube_simple.blend","./tests/test_data/cube.stl"])
def test_geometry_loading(path):

    walls = bh.read_geometry_file(path, auto_walls=True)

    assert walls["conn"].shape[0]==6
    assert walls["normal"].shape[0]==walls["conn"].shape[0]
    assert walls["verts"].shape[0]==8
    assert walls["normal"].shape[1]==3

def test_auto_wall():
    """Test if auto/manual wall assignment is working as expected."""
    # check if sub-walls of same material are merged
    # or individually picked up correctly
    simple_merge = bh.read_geometry_file("./tests/test_data/cube_simple.blend",
                                          auto_walls=True)
    simple_split = bh.read_geometry_file("./tests/test_data/cube_simple.blend",
                                          auto_walls=False)

    assert simple_merge["conn"].shape[0]!=simple_split["conn"].shape[0]
    assert simple_split["conn"].shape[0]==24

    # check if sub-walls of different material are merged
    # or individually picked up correctly
    dual_merge = bh.read_geometry_file("./tests/test_data/cube_subdiv.blend",
                                       auto_walls=True)
    dual_split = bh.read_geometry_file("./tests/test_data/cube_subdiv.blend",
                                       auto_walls=False)

    assert dual_merge["conn"].shape[0]==10
    npt.assert_array_equal(dual_merge["conn"],dual_split["conn"])

@pytest.mark.parametrize("blend_file",
                         ["./tests/test_data/cube.stl","./tests/test_data/cube_simple.blend"])
def test_patch_to_wall_maps(blend_file):
    """Check if patch/wall connectivity makes sense."""
    walls = bh.read_geometry_file(blend_file=blend_file, auto_walls=True)
    patches = bh.read_geometry_file(blend_file=blend_file, auto_walls=False)

    patches["wallID"] =np.empty((patches["conn"].shape[0],),dtype=int)

    for i,pconn in enumerate(patches["conn"]):
        pt = patches["verts"][pconn][0]
        for j,wconn in enumerate(walls["conn"]):
            if (patches["normal"][i]==walls["normal"][j]).all():
                if patches["material"][i]==walls["material"][j]:
                    if point_in_polygon(point3d=pt,
                                        polygon3d=walls["verts"][wconn],
                                        plane_normal=walls["normal"][j]):
                        patches["wallID"][i]=j

    print("hey hey hey")

@pytest.mark.parametrize("path",
                         ["./tests/test_data/cube_simple.blend","./tests/test_data/cube.stl"])
def test_material_assignment(path):
    """Check that material is correctly assigned from model."""
    walls = bh.read_geometry_file(path)
    Acount=0
    Bcount=0
    if path.endswith(".blend"):
        assert len(walls["material"])==len(walls["conn"])
        assert isinstance(walls["material"][0], str)
        for i in range(len(walls["conn"])):
            if walls["material"][i]=="matA":
                Acount+=1
            elif walls["material"][i]=="matB":
                Bcount+=1
        assert Acount==1
        assert Bcount==5
    else:
        assert (walls["material"]=="").all()
    assert True
