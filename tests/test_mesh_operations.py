import numpy as np
import numpy.testing as npt
import pytest
import os
import pyfar as pf

import sparapy as sp
from sparapy.radiosity_fast import blender_helpers as bh

@pytest.mark.parametrize("path", ["./tests/test_data/cube.blend","./tests/test_data/cube.stl"])
def test_mesh_simplification(path):

    fine, rough = bh.read_geometry_file(path)

    npt.assert_array_equal(fine["verts"],rough["verts"])
    assert len(fine["conn"]) == len(rough["conn"])*2
