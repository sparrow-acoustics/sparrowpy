import numpy.testing as npt
import pytest
import numpy as np
import sparrowpy.utils.blender as bh

@pytest.mark.parametrize("path",
                         ["./tests/test_data/cube_simple.blend","./tests/test_data/cube.stl"])
def test_geometry_loading(path):

    ## check that individual walls are extracted from file

    ## check that auto_walls are correctly generated
    geom_w = bh.read_geometry_file(path, auto_walls=True,
                                 patches_from_model=False)

    assert not geom_w["patch"]
    assert geom_w["wall"]["conn"].shape[0]==6
    assert geom_w["wall"]["normal"].shape[0]==geom_w["wall"]["conn"].shape[0]
    assert geom_w["wall"]["verts"].shape[0]==8
    assert geom_w["wall"]["normal"].shape[1]==3


    geom_wp= bh.read_geometry_file(path, auto_walls=True,
                                 patches_from_model=True)


    assert bool(geom_wp["patch"])
    assert type(geom_wp["wall"]["conn"]) is list
    npt.assert_equal(np.array(geom_wp["wall"]["conn"]),
                            geom_w["wall"]["conn"])
    npt.assert_equal(geom_wp["wall"]["normal"],
                            geom_w["wall"]["normal"])
    npt.assert_equal(geom_wp["wall"]["verts"],
                            geom_w["wall"]["verts"])
    npt.assert_equal(geom_wp["wall"]["material"],
                            geom_w["wall"]["material"])

    if path.endswith(".blend"):
        assert geom_wp["patch"]["conn"].shape[0]==24
        assert geom_wp["patch"]["verts"].shape[0]==26
    else:
        assert geom_wp["patch"]["conn"].shape[0]==12
        assert geom_wp["patch"]["verts"].shape[0]==8




@pytest.mark.parametrize("path",
                         ["./tests/test_data/cube_simple.blend","./tests/test_data/cube.stl"])
def test_material_assignment(path):
    """Check that material is correctly assigned from model."""
    geom = bh.read_geometry_file(path)
    walls= geom["wall"]
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
