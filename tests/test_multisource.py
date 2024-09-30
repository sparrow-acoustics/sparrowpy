"""Test multiple source capabilities."""
import os

import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
import sparapy as sp
import matplotlib.pyplot as plt
import numpy as np

create_reference_files = False


@pytest.mark.parametrize('src', [
    [[2.,0,0], [-1, 0, 0], [0, 0, 1]],
    [[2.,2.,0], [-1, 0, 0], [0, 0, 1]],
    [[2.,0.,2.], [-1, 0, 0], [0, 0, 1]],
    ])
@pytest.mark.parametrize('rec', [
    [[1.,0.,0], [-1, 0, 0], [0, 0, 1]],
    [[2.,-2.,0], [-1, 0, 0], [0, 0, 1]],
    [[2.,0.,-2.], [-1, 0, 0], [0, 0, 1]],
    ])
def test_reciprocity_s2p_p2r(src,rec):
    """check if radiosity implementation has source-receiver reciprocity"""
    wall = [sp.geometry.Polygon(
            [[0, -1, -1], [0, -1, 1],
            [0, 1, 1], [0, 1, -1]],
            [0, 0, 1], [1, 0, 0])]
    
    energy=[]

    for i in range(2):
        if i == 0:
            src_ = sp.geometry.SoundSource(src[0],src[1], src[2])
            rec_ = sp.geometry.Receiver(rec[0],rec[1], rec[2])
        elif i == 1:
            src_ = sp.geometry.SoundSource(rec[0],rec[1], rec[2])
            rec_ = sp.geometry.Receiver(src[0],src[1], src[2])

        e = sp.form_factor.pt_solution(point=src_.position,patch_points=wall[0].pts)

        e *= sp.form_factor.pt_solution(point=rec_.position,patch_points=wall[0].pts)

        energy.append(e)

    
    npt.assert_array_almost_equal(energy[0], energy[1])
