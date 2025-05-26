import numpy.testing as npt
import pytest
bpy = pytest.importorskip("bpy")
import sparrowpy.utils.blender as bh  # noqa: E402
import numpy as np # noqa: E402

@pytest.mark.parametrize("path",
                         ["./tests/test_data/cube_simple.blend","./tests/test_data/cube.stl"])
def test_geometry_loading(path):
    """Test that geometry data is correctly loaded from file."""
    geom_w = bh.read_geometry_file(path, wall_auto_assembly=True)

    assert len(geom_w["conn"])==6
    assert geom_w["normal"].shape[0]==len(geom_w["conn"])
    assert geom_w["up"].shape[0]==len(geom_w["conn"])
    assert geom_w["verts"].shape[0]==8
    assert geom_w["normal"].shape[1]==3
    assert geom_w["up"].shape[1]==3
    for i in range(len(geom_w["conn"])):
        npt.assert_almost_equal(np.inner(geom_w["up"][i],geom_w["normal"][i]),0)


    geom_p= bh.read_geometry_file(path, wall_auto_assembly=False)

    if path.endswith(".blend"):
        assert len(geom_p["conn"])==24
        assert geom_p["verts"].shape[0]==26
    else:
        assert len(geom_p["conn"])==12
        assert geom_p["verts"].shape[0]==8




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
        assert walls["material"] is None
