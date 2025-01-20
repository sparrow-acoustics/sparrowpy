import numpy.testing as npt
import pytest

import sparrowpy.utils.blender as bh

@pytest.mark.parametrize("path",
                         ["./tests/test_data/cube.blend","./tests/test_data/cube.stl"])
def test_mesh_simplification(path):

    fine, rough = bh.read_geometry_file(path)

    npt.assert_array_equal(fine["verts"],rough["verts"])
    assert len(fine["conn"]) == len(rough["conn"])*2
    assert len(fine["norm"]) == len(rough["norm"])*2
    assert fine["norm"].shape[1]==rough["norm"].shape[1]
    assert fine["norm"].shape[1]==3
