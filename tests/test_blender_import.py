import numpy.testing as npt
import pytest
bpy = pytest.importorskip("bpy")
import sparrowpy.utils.blender as bh  # noqa: E402
import numpy as np # noqa: E402
import trimesh as tm

@pytest.mark.parametrize("path",
                         ["./tests/test_data/cube_simple.blend","./tests/test_data/cube.stl"])
def test_geometry_loading(path):
    """Test that geometry data is correctly loaded from file."""
    geom_w = bh.read_geometry_file(path, wall_auto_assembly=True,
                                 patches_from_model=False)

    assert not geom_w["patch"]
    assert geom_w["wall"]["conn"].shape[0]==6
    assert geom_w["wall"]["normal"].shape[0]==geom_w["wall"]["conn"].shape[0]
    assert geom_w["wall"]["up"].shape[0]==geom_w["wall"]["conn"].shape[0]
    assert geom_w["wall"]["verts"].shape[0]==8
    assert geom_w["wall"]["normal"].shape[1]==3
    assert geom_w["wall"]["up"].shape[1]==3
    for i in range(geom_w["wall"]["conn"].shape[0]):
        npt.assert_almost_equal(np.inner(geom_w["wall"]["up"][i],geom_w["wall"]["normal"][i]),0)


    geom_wp= bh.read_geometry_file(path, wall_auto_assembly=True,
                                 patches_from_model=True)


    assert bool(geom_wp["patch"])
    assert isinstance(geom_wp["wall"]["conn"], list)
    npt.assert_equal(np.array(geom_wp["wall"]["conn"]),
                            geom_w["wall"]["conn"])
    npt.assert_equal(geom_wp["wall"]["normal"],
                            geom_w["wall"]["normal"])
    npt.assert_equal(geom_wp["wall"]["verts"],
                            geom_w["wall"]["verts"])
    npt.assert_equal(geom_wp["wall"]["material"],
                            geom_w["wall"]["material"])

    if path.endswith(".blend"):
        assert geom_wp["patch"]["conn"].shape[0]==2*24
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

@pytest.mark.parametrize("path",
                         ["./tests/test_data/ico.blend"])
def test_point_cloud(path):
    """Check that patches can be generated from point clouds."""
    geom = bh.read_geometry_file(path,blender_geom_id="Icosphere",
                                 patch_geom_id="Cube")

    tm.util.attach_to_log()

    mesh = tm.Trimesh(vertices=geom["patch"]["verts"],
                      faces=geom["patch"]["conn"])

    mesh.show(flags={'wireframe': True})

