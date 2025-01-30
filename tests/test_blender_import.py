import numpy.testing as npt
import pytest

import sparrowpy.utils.blender as bh

@pytest.mark.parametrize("path",
                         ["./tests/test_data/cube.blend","./tests/test_data/cube.stl"])
def test_mesh_simplification(path):

    walls,patches = bh.read_geometry_file(path)


