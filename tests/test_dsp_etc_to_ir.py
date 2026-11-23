import pytest
import numpy as np
import pyfar as pf
from sparrowpy import dsp
import numpy.testing as npt


@pytest.mark.parametrize("delay_samples", [0, 440])
def test_etc_from_ir_impulse(delay_samples):
    n_samples = 44100
    impulse_response = pf.signals.impulse(n_samples, delay=delay_samples)
    etc = dsp.energy_time_curve_from_impulse_response(
        impulse_response, delta_time=0.01)
    assert isinstance(etc, pf.TimeData)
    assert etc.n_samples == 100
    npt.assert_allclose(etc.time[:, 0], 1)
    npt.assert_allclose(etc.time[:, 1:], 0)


@pytest.mark.parametrize("delay_samples", [0, 440])
@pytest.mark.parametrize("bw_factor", [1, 2, 4])
def test_etc_from_ir_impulse_bandwidth(delay_samples, bw_factor):
    n_samples = 44100
    bw = n_samples/2/bw_factor

    impulse_response = pf.signals.impulse(n_samples, delay=delay_samples)
    etc = dsp.energy_time_curve_from_impulse_response(
        impulse_response, delta_time=0.01, bandwidth=bw)
    assert isinstance(etc, pf.TimeData)
    assert etc.n_samples == 100
    npt.assert_allclose(etc.time[:, 0], bw_factor)
    npt.assert_allclose(etc.time[:, 1:], 0)


@pytest.mark.parametrize("delta_time", [0.1, 0.01, 0.001])
def test_etc_from_ir_impulse_delta_time(delta_time):
    n_samples = 44100
    impulse_response = pf.signals.impulse(n_samples, delay=0)
    etc = dsp.energy_time_curve_from_impulse_response(
        impulse_response, delta_time=delta_time)
    assert isinstance(etc, pf.TimeData)
    assert etc.times[1]-etc.times[0] == delta_time
    npt.assert_allclose(etc.time[:, 0], 1)
    npt.assert_allclose(etc.time[:, 1:], 0)


@pytest.mark.parametrize("delay_samples", [0, 440])
@pytest.mark.parametrize("bw_factor", [1, 2, 4])
def test_etc_from_ir_impulse_multichannel(delay_samples, bw_factor):
    n_samples = 44100
    bw = np.ones(2)*n_samples/2/bw_factor

    impulse_response = pf.signals.impulse(
        n_samples, delay=delay_samples, amplitude=np.ones((2, 2))*0.5)
    etc = dsp.energy_time_curve_from_impulse_response(
        impulse_response, delta_time=0.01, bandwidth=bw)
    assert isinstance(etc, pf.TimeData)
    assert etc.n_samples == 100
    npt.assert_allclose(etc.time[..., 0], 0.25*bw_factor)
    npt.assert_allclose(etc.time[..., 1:], 0)


def test_etc_from_ir_pulsed_dirac():
    n_samples = 44100
    dirac = pf.signals.impulse(
        n_samples, delay=np.arange(0, n_samples, 441))*0.5
    pulsed_dirac = pf.Signal(
        np.sum(dirac.time, axis=0, keepdims=True),
        dirac.sampling_rate,
    )
    etc = dsp.energy_time_curve_from_impulse_response(
        pulsed_dirac, delta_time=0.01)
    assert isinstance(etc, pf.TimeData)
    assert etc.n_samples == 100
    npt.assert_allclose(etc.time, 0.25)


def test_etc_from_ir_filtered_noise():
    n_samples = 44100
    white_noise = pf.signals.noise(n_samples, "white", seed=0)
    _, _, cutoff = pf.dsp.filter.fractional_octave_frequencies(
        1, (1e3, 22e3), return_cutoff=True)
    bandwidth = cutoff[1] - cutoff[0]
    filtered_noise = pf.dsp.filter.fractional_octave_bands(
        white_noise, num_fractions=1, frequency_range=(1e3, 22e3))
    filtered_noise.time = np.swapaxes(filtered_noise.time,
                                      0,
                                      filtered_noise.cdim-1)
    etc = dsp.energy_time_curve_from_impulse_response(
        filtered_noise, delta_time=0.01, bandwidth=bandwidth)
    assert isinstance(etc, pf.TimeData)
    assert etc.n_samples == 100
    assert np.any(etc.time > 0)
    # test if energy is roughly same for white noise bands
    npt.assert_almost_equal(
        10*np.log10(np.mean(etc.time, axis=-1)), 26, decimal=0)


def test_etc_from_ir_type_error_signal():
    with pytest.raises(
            TypeError, match="signal must be a pyfar Signal object."):
        dsp.energy_time_curve_from_impulse_response(np.array([1, 2, 3]))


def test_etc_from_ir_type_error_delta_time():
    signal = pf.signals.impulse(10)
    with pytest.raises(TypeError, match="delta_time must be a float or int."):
        dsp.energy_time_curve_from_impulse_response(signal, delta_time="0.01")


def test_etc_from_ir_value_error_delta_time():
    signal = pf.signals.impulse(10)
    with pytest.raises(ValueError, match="delta_time must be positive."):
        dsp.energy_time_curve_from_impulse_response(signal, delta_time=0)


def test_etc_from_ir_value_error_bandwidth_negative():
    signal = pf.signals.impulse(10)
    with pytest.raises(ValueError, match="bandwidth must be positive."):
        dsp.energy_time_curve_from_impulse_response(signal, bandwidth=-1)


def test_etc_from_ir_value_error_bandwidth_array_negative():
    signal = pf.signals.impulse(10)
    bw = np.array([-1, 2])
    # Adjust signal to have 2 bands
    signal = pf.Signal(np.ones((2, 10)), 44100)
    with pytest.raises(
            ValueError, match="All bandwidth values must be positive."):
        dsp.energy_time_curve_from_impulse_response(signal, bandwidth=bw)


def test_etc_from_ir_value_error_bandwidth_shape():
    signal = pf.Signal(np.ones((2, 10)), 44100)
    bw = np.ones(3)
    with pytest.raises(ValueError, match="bandwidth shape"):
        dsp.energy_time_curve_from_impulse_response(signal, bandwidth=bw)
