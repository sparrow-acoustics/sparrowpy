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
    [[2.,1.,1.], [0, 1, 0], [0, 0, 1]],
    [[.5,0.5,.5], [-1, 1, 0], [0, 0, 1]],
    [[.5,2,2.5], [1, 0, 0], [0, 0, 1]]
    ])
@pytest.mark.parametrize('rec', [
    [[1.,1.,.5], [-1, 1, 0], [0, 0, 1]],
    ])
@pytest.mark.parametrize('ord', [
    1,5,10,20
    ])
@pytest.mark.parametrize('ps', [
    1.5,3
    ])
def test_reciprocity_shoebox(src,rec,ord,ps):
    """Test if the results are reciprocal in shoebox room."""
    X = 3
    Y = 3
    Z = 3
    patch_size = ps
    ir_length_s = 1
    sampling_rate = 100
    max_order_k = ord
    speed_of_sound = 343
    irs_new = []
    frequencies = np.array([1000])
    absorption = 0.
    walls = sp.testing.shoebox_room_stub(X, Y, Z)
    method= "universal"
    algo= "order"

    sc_src = pf.Coordinates(0, 0, 1)
    sc_rec = pf.Coordinates(0, 0, 1)

    for i in range(2):
        if i == 0:
            src_ = sp.geometry.SoundSource(src[0],src[1], src[2])
            rec_ = sp.geometry.Receiver(rec[0],rec[1], rec[2])
        elif i == 1:
            src_ = sp.geometry.SoundSource(rec[0],rec[1], rec[2])
            rec_ = sp.geometry.Receiver(src[0],src[1], src[2])


        ## initialize radiosity class
        radi = sp.radiosity_fast.DRadiosityFast.from_polygon(walls, patch_size)

        data_scattering = pf.FrequencyData(
            np.ones((sc_src.csize, sc_rec.csize, frequencies.size)), frequencies)

        # set directional scattering data
        radi.set_wall_scattering(
            np.arange(len(walls)), data_scattering, sc_src, sc_rec)

        # set air absorption
        radi.set_air_attenuation(
            pf.FrequencyData(
                np.zeros_like(data_scattering.frequencies),
                data_scattering.frequencies))

        # set absorption coefficient
        radi.set_wall_absorption(
            np.arange(len(walls)),
            pf.FrequencyData(
                np.zeros_like(data_scattering.frequencies)+absorption,
                data_scattering.frequencies))

        # run simulation
        radi.bake_geometry(ff_method=method, algorithm=algo)

        radi.init_source_energy(src_.position, ff_method=method, algorithm=algo)

        ir = radi.calculate_energy_exchange_receiver(
                receiver_pos=rec_.position, speed_of_sound=speed_of_sound,
                histogram_time_resolution=1/sampling_rate,
                histogram_length=ir_length_s, ff_method=method, algorithm=algo,
                max_depth=max_order_k )

        # test energy at receiver
        irs_new.append(ir)

    irs_new = np.array(irs_new)

    npt.assert_array_almost_equal(np.sum(irs_new[1]), np.sum(irs_new[0]))
    npt.assert_array_almost_equal(irs_new[1][0], irs_new[0][0])


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

        e = sp.form_factor.pt_solution(point=src_.position,patch_points=wall[0].pts, mode="source")

        e *= sp.form_factor.pt_solution(point=rec_.position,patch_points=wall[0].pts, mode="receiver")

        energy.append(e)

    
    npt.assert_array_almost_equal(energy[0], energy[1])
