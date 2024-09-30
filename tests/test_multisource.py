"""Test multiple source capabilities."""
import os

import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
import sparapy as sp
import sparapy.geometry as geo
import sparapy.radiosity as radiosity
from sparapy.sound_object import Receiver, SoundSource
import matplotlib.pyplot as plt

create_reference_files = False

@pytest.mark.parametrize('src', [
    [[2.,1.,1.], [0, 1, 0], [0, 0, 1]],
    [[.5,0.5,.5], [-1, 1, 0], [0, 0, 1]],
    [[.5,2,2.5], [1, 0, 0], [0, 0, 1]]
    ])
@pytest.mark.parametrize('rec', [
    [[1.,2,1], [-1, 0, 0], [0, 0, 1]]
    ])
def test_reciprocity(src,rec):
    """Test if the results changes for shifted walls."""
    X = 3
    Y = 3
    Z = 3
    patch_size = .5
    ir_length_s = 1
    sampling_rate = 1000
    max_order_k = 10
    speed_of_sound = 343
    irs_new = []
    E_matrix = []
    form_factors = []

    delta_x = 0.
    delta_y = 0.
    delta_z = 0.

    for i in range(2):
        walls = sp.testing.shoebox_room_stub(X, Y, Z)
        if i == 0:
            src_ = sp.geometry.SoundSource(src[0],src[1], src[2])
            rec_ = sp.geometry.Receiver(rec[0],rec[1], rec[2])
        elif i == 1:
            src_ = sp.geometry.SoundSource(rec[0],rec[1], rec[2])
            rec_ = sp.geometry.Receiver(src[0],src[1], src[2])

        for wall in walls:
            wall.pts += np.array([delta_x, delta_y, delta_z])


        ## new approach
        radi = sp.radiosity.Radiosity(
            walls, patch_size, max_order_k, ir_length_s,
            speed_of_sound=speed_of_sound, sampling_rate=sampling_rate,
            absorption=0.1)

        # run simulation
        radi.run(src_)

        E_matrix.append(np.concatenate([
            radi.patch_list[i].E_matrix for i in range(6)], axis=-2))
        form_factors.append([
            radi.patch_list[i].form_factors for i in range(6)])

        # test energy at receiver
        irs_new.append(radi.energy_at_receiver(rec_, ignore_direct=False))

    irs_new = np.array(irs_new)

    fig,ax=plt.subplots()
    ax.plot(irs_new[0][0][irs_new[0][0]>10**-6], "o")
    ax.plot(irs_new[1][0][irs_new[1][0]>10**-6],"*")
    plt.show()

    # rotate all walls
    
    npt.assert_array_almost_equal(irs_new[1], irs_new[0])


