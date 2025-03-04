import sparrowpy as sp
import pyfar as pf #type: ignore
import numpy as np
import pytest

def test_dsp_():
    speed_of_sound = 343

    X = 4
    Y = 5
    Z = 3
    room_volume = X * Y * Z

    ir_length_s = 1
    sampling_rate = 8000    # 1/delta_t (below 48kHz)
    patch_size = 1
    max_order_k = 140

    hist_data = sp.dsp.run_energy_simulation_room(
        dimensions=[X, Y, Z],
        patch_size=patch_size,
        ir_length_s=ir_length_s,
        sampling_rate=sampling_rate,
        max_order_k=max_order_k,
        source=pf.Coordinates(1, 1, 1),
        receiver=pf.Coordinates(2.5, 3.5, 1.6),
        speed_of_sound=speed_of_sound)

    assert isinstance(hist_data, np.ndarray)
    assert hist_data.ndim == 4

    # sum over all patches
    hist_sum = np.sum(hist_data[0, :, 0, :], axis=0)
    assert hist_sum.ndims == 1
    hist_sig = pf.Signal(hist_sum, sampling_rate)

    ### TODO: CHECK histogram_resolution_reduction

    dirac_sig = sp.dsp.generate_dirac_sequence_raven(
        room_volume=room_volume,
        speed_of_sound=speed_of_sound,
        ir_length_s_stop=ir_length_s)
    assert isinstance(dirac_sig, pf.Signal)
    assert dirac_sig.sampling_rate == 48000
    assert dirac_sig.time.shape
