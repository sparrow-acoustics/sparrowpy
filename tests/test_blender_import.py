import numpy.testing as npt
import pytest
bpy = pytest.importorskip("bpy")
import sparrowpy.utils.blender as bh  # noqa: E402


@pytest.mark.parametrize("path",
                         ["./tests/test_data/cube.blend","./tests/test_data/cube.stl"])
def test_mesh_simplification(path):

    fine, rough = bh.read_geometry_file(path)

    npt.assert_array_equal(fine["verts"],rough["verts"])
    assert len(fine["conn"]) == len(rough["conn"])*2
