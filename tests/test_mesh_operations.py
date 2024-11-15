import numpy as np
import numpy.testing as npt
import pytest
import os
import pyfar as pf

import sparapy as sp
from sparapy.radiosity_fast import blender_helpers as bh


def test_mesh_simplification():

    fine, rough = bh.read_blend_file("./tests/test_data/cube.blend")

    npt.assert_array_equal(fine["verts"],rough["verts"])
    assert len(fine["conn"]) == len(rough["conn"])*2
