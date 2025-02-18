import numpy.testing as npt
import pytest
import numpy as np
import sparrowpy.utils.blender as bh

@pytest.mark.parametrize("path",
                         ["./tests/test_data/cube_simple.blend","./tests/test_data/cube.stl"])
def test_geometry_loading(path):

    geom = bh.read_geometry_file(path, auto_walls=True,
                                 patches_from_model=True)

    assert geom["wall"]["conn"].shape[0]==6
    assert geom["wall"]["normal"].shape[0]==geom["wall"]["conn"].shape[0]
    assert geom["wall"]["verts"].shape[0]==8
    assert geom["wall"]["normal"].shape[1]==3


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
