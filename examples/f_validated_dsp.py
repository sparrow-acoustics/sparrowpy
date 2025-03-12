import sparrowpy as sp
import pyfar as pf #type: ignore
import numpy as np
# import pytest
# ruff: noqa: FIX002 ERA001 D103 D100

def test_dsp_walkthrough() -> tuple[pf.Signal, pf.Signal, pf.Signal, pf.Signal]:
    speed_of_sound = 343

    X = 4
    Y = 5
    Z = 3
    room_volume = X * Y * Z

    ir_length_s = 1
    sampling_rate = 4000    # 1/delta_t (below 48kHz)
    patch_size = 1
    max_order_k = 100

    hist_data = sp.dsp.run_energy_simulation_room(
        dimensions=[X, Y, Z],
        patch_size=patch_size,
        ir_length_s=ir_length_s,
        sampling_rate=sampling_rate,
        max_order_k=max_order_k,
        source=pf.Coordinates(1, 1, 1),
        receiver=pf.Coordinates(2.5, 3.5, 1.6),
        speed_of_sound=speed_of_sound,
    )
    assert isinstance(hist_data, np.ndarray)
    assert hist_data.ndim

    # sum over all patches
    hist_sum = np.sum(hist_data[0, :, 0, :], axis=0)
    assert hist_sum.ndim
    hist_sig = pf.Signal(hist_sum, sampling_rate)

    dirac_sig = sp.dsp.generate_dirac_sequence_raven(
        room_volume=room_volume,
        speed_of_sound=speed_of_sound,
        ir_length_s_stop=ir_length_s,
        sampling_rate_dirac=48000)
    assert isinstance(dirac_sig, pf.Signal)
    assert dirac_sig.sampling_rate == 48000
    print(dirac_sig.time.shape)
    print(dirac_sig.time.ndim)

    IR_sig = sp.dsp.dirac_weighted_no_filter(
        dirac_sig, hist_sig)
    assert isinstance(IR_sig, pf.Signal)

    dirac_filt_sig, centerFreq = sp.dsp.dirac_band_filter(dirac_sig)
    assert isinstance(dirac_filt_sig, pf.Signal) #through typeignore?
    assert centerFreq is not None
    print(dirac_filt_sig.time.shape)

    IR_bands_sig, IR_sum_full_sig = sp.dsp.dirac_weighted_with_filter(
        dirac_filt_sig, centerFreq, hist_sig)
    assert isinstance(IR_bands_sig, pf.Signal)
    assert isinstance(IR_sum_full_sig, pf.Signal)

    return hist_sig, IR_sig, IR_bands_sig, IR_sum_full_sig


def test_histogram_reduction(
    hist_sig,
    sampling_rate_new,
    ) -> pf.Signal:
    sine_sig = pf.signals.sine(
        frequency=1000,
        n_samples=96000,
        amplitude=1,
        sampling_rate=48000)
    noise_white_sig = pf.signals.noise(
        n_samples=96000,
        spectrum='white',
        rms=2,
        sampling_rate=48000)
    noise_pink_sig = pf.signals.noise(
        n_samples=96000,
        spectrum="pink",
        rms=2,
        sampling_rate=48000)
    #energy sum of all samples
    #power 1/n and rms (1/n)^2
    assert abs(pf.dsp.rms(sine_sig) - 1/np.sqrt(2)) < 0.01
    assert abs(pf.dsp.rms(noise_white_sig) - 2) < 0.01
    assert abs(pf.dsp.rms(noise_pink_sig) - 2) < 0.01
    assert abs(pf.dsp.power(sine_sig) - 1/np.sqrt(2)/np.sqrt(2)) < 0.01
    assert abs(pf.dsp.power(noise_pink_sig) - np.sqrt(2)) < 0.01
    assert abs(pf.dsp.power(noise_pink_sig) - np.sqrt(2)) < 0.01

    hist_red_sig = sp.dsp.histogram_resolution_reduction(hist_sig, sampling_rate_new)
    ##assert hist_red_sig.sampling_rate == sampling_rate_new
    ##assert pf.dsp.energy(hist_red_sig) == pf.dsp.energy(hist_sig)
    ##assert pf.dsp.power(hist_red_sig) == pf.dsp.power(hist_sig)
    ##assert pf.dsp.rms(hist_red_sig) == pf.dsp.rms(hist_sig)
    return hist_red_sig
