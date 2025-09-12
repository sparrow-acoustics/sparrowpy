import pytest
import numpy as np
import numpy.testing as npt
import sparrowpy as sp
import pyfar as pf
import matplotlib.pyplot as plt

@pytest.mark.parametrize("room_volume", [200, 500])
@pytest.mark.parametrize("speed_of_sound", [320, 343])
@pytest.mark.parametrize("n_samples", [22050, 44100])
@pytest.mark.parametrize("max_reflection_density", [None, 5000])
def test_reflection_density_room(
        n_samples, speed_of_sound, room_volume, max_reflection_density):
    reflection_density, t_start = sp.dsp.reflection_density_room(
        room_volume, n_samples, speed_of_sound=speed_of_sound,
        max_reflection_density=max_reflection_density)

    # test reflection_density
    assert isinstance(reflection_density, pf.TimeData)
    t = reflection_density.times
    desired = 4*np.pi*speed_of_sound**3*t**2/room_volume
    if max_reflection_density is not None:
        desired[desired > max_reflection_density] = max_reflection_density
    npt.assert_almost_equal(
        reflection_density.time.flatten(), desired)

    # test t_start
    t_desired = (2*room_volume*np.log(2)/(4*np.pi*speed_of_sound**3))**(1/3)
    npt.assert_almost_equal(t_start, t_desired, decimal=3)


def test_reflection_density_room_inputs():
    """Test that inputs must be positive."""
    with pytest.raises(ValueError, match="speed_of_sound must be positive."):
        sp.dsp.reflection_density_room(
            room_volume=500, n_samples=44100, speed_of_sound=-343)

    with pytest.raises(ValueError, match="room_volume must be positive."):
        sp.dsp.reflection_density_room(
            room_volume=-500, n_samples=44100)

    with pytest.raises(ValueError, match="n_samples must be positive."):
        sp.dsp.reflection_density_room(
            room_volume=500, n_samples=-44100)

    with pytest.raises(
            ValueError, match="max_reflection_density must be positive."):
        sp.dsp.reflection_density_room(
            room_volume=500, n_samples=44100, max_reflection_density=-10000)


@pytest.mark.parametrize("t_start", [0, 0.01])
@pytest.mark.parametrize("n_samples", [2205, 4410])
@pytest.mark.parametrize("sampling_rate", [4410, 4800])
def test_dirac_sequence_dirac(
        t_start, n_samples, sampling_rate):
    times = pf.Signal(np.zeros(n_samples), sampling_rate).times
    reflection_density = pf.TimeData(times**2*1e3+100, times)
    sequence = sp.dsp.dirac_sequence(
        reflection_density, n_samples=n_samples, t_start=t_start,
        sampling_rate=sampling_rate,
        )

    # test if the sequence is a Signal object
    assert isinstance(sequence, pf.Signal)
    assert sequence.sampling_rate == sampling_rate

    # test if the sequence is a 1D array
    assert sequence.cdim == 1
    values = set(np.round(sequence.time.flatten(), 3))
    for value in values:
        assert value in {0, 1, -1}, f"Unexpected value in sequence: {value}"
    assert np.sum(np.abs(sequence.time)) < n_samples

    # test if the density is higher at the end of the ir
    n_center = int(n_samples/2)
    assert np.sum(np.abs(sequence.time[..., :n_center])) < np.sum(np.abs(
        sequence.time[..., n_center:]))

    # test if no dirac before t_start
    n_start = int(t_start * sequence.sampling_rate)
    assert np.sum(np.abs(sequence.time[..., :n_start])) == 0


@pytest.mark.parametrize("n_samples", [22050, 44100])
@pytest.mark.parametrize("sampling_rate", [44100, 48000])
def test_dirac_sequence_constant_reflection_density(
        n_samples, sampling_rate):
    mu = 200
    times = pf.Signal(np.zeros(n_samples), sampling_rate).times
    reflection_density = pf.TimeData(np.ones(n_samples)*mu, times)
    t_start = 0
    sequence = sp.dsp.dirac_sequence(
        reflection_density, n_samples, t_start,
        sampling_rate=sampling_rate, seed=2,
        )

    # test if the sequence is a Signal object
    assert isinstance(sequence, pf.Signal)
    assert sequence.sampling_rate == sampling_rate

    # test if the sequence is a 1D array
    assert sequence.cdim == 1
    values = set(np.round(sequence.time.flatten(), 3))
    for value in values:
        assert value in {0, 1, -1}, f"Unexpected value in sequence: {value}"
    assert np.sum(np.abs(sequence.time)) < n_samples

    # test delta_time
    delta_times = []
    delta_time = 0
    for i in range(sequence.n_samples):
        if sequence.time[..., i] == 0:
            delta_time += 1
        else:
            delta_times.append(delta_time/sequence.sampling_rate)
            delta_time = 0

    npt.assert_almost_equal(1/np.mean(delta_times)/mu, 1, decimal=1)


def test_dirac_sequence_inputs():
    """Test that inputs must be positive."""
    with pytest.raises(
            ValueError,
            match="reflection_density must be a pyfar.TimeData object."):
        sp.dsp.dirac_sequence(500, 400)

    with pytest.raises(ValueError, match="t_start must be positive."):
        sp.dsp.dirac_sequence(
            pf.TimeData([0], [0]), 500, -400)

    with pytest.raises(
            ValueError,
            match="The reflection density must be less than sampling_rate/2."):
        sp.dsp.dirac_sequence(
            pf.TimeData([44100], [0]), 500)


@pytest.mark.parametrize("sr", [
    44100, 48000,
    ])
@pytest.mark.parametrize("etc_step",[
    1/441, 1/500,
])
def test_etc_weighting_broadspectrum(sr,etc_step):
    """Test that broadband noise is correctly weighted by the etc."""
    noise = pf.signals.noise(sampling_rate=sr,n_samples=.1*sr)
    times = np.arange(0,.1,etc_step)

    etc = pf.TimeData(np.random.rand(times.shape[0]), times)

    sig = sp.dsp.weight_filters_by_etc(etc=etc,
                                       noise_filters=noise,
                                       )

    etc_from_sig = sp.dsp.energy_time_curve_from_impulse_response(
        signal=sig,
        delta_time=etc_step,
    )
    npt.assert_allclose(etc.time,etc_from_sig.time)

@pytest.mark.parametrize("n_receivers", [
    0,1,2,
    ])
@pytest.mark.parametrize("n_directions",[
    0,1,2,
])
@pytest.mark.parametrize("n_freqs",[
    1,2,3,
])
@pytest.mark.parametrize("bw_depth",[
    1,2,3,
])
def test_etc_weighting_multichannel(n_receivers,n_directions,n_freqs,bw_depth):
    """Test that channel handling of etc weighting is done correctly."""
    sr = 100
    delta=1/sr*10
    times = np.arange(0,1,delta)
    data = np.random.rand(n_freqs,times.shape[0])

    if n_directions>0:
        data = data[np.newaxis,:]
        data = np.repeat(data,n_directions,axis=0)

    if n_receivers>0:
        data = data[np.newaxis,:]
        data = np.repeat(data,n_receivers,axis=0)

    etc = pf.TimeData(data, times)

    bandwidths = np.random.rand(*etc.cshape[-bw_depth:])**2 * sr/4

    noise = pf.signals.noise(n_samples=sr, sampling_rate=sr)
    noise.time=np.broadcast_to(noise.time,bandwidths.shape+(noise.time.shape[-1],))

    sig = sp.dsp.weight_filters_by_etc(etc=etc,
                                       noise_filters=noise,
                                       bandwidth=bandwidths,
                                       )

    assert(sig.cshape==etc.cshape)

    etc_from_sig = sp.dsp.energy_time_curve_from_impulse_response(
        signal=sig,
        delta_time=delta,
        bandwidth=bandwidths,
    )

    npt.assert_allclose(etc.time,etc_from_sig.time)

def test_etc_weighting_inputs():
    """Test that inputs respect formatting."""
    noise_filters = pf.signals.noise(n_samples=100)
    noise_filters.time = np.repeat(noise_filters.time,2,axis=0)
    etc = pf.TimeData(np.array([[1,1],[1,1]]),times=np.array([0,0.001]))

    with pytest.raises(
            ValueError,
            match="Bandwidth must be positive."):
        sp.dsp.weight_filters_by_etc(etc=etc,noise_filters=noise_filters,bandwidth=-1)

    with pytest.raises(
            ValueError,
            match="All bandwidth values must be positive."):
        sp.dsp.weight_filters_by_etc(etc=etc,noise_filters=noise_filters,
                                     bandwidth=[50,-30])

    etc = pf.TimeData(np.array([[1,1,1],[1,1,1]]),
                      times=np.array([0,0.001,0.2]))

    with pytest.raises(
            ValueError,
            match="ETC entries must be equally spaced in time."):
        sp.dsp.weight_filters_by_etc(etc=etc,noise_filters=noise_filters,
                                     bandwidth=[50,50])



@pytest.mark.parametrize("freq",[
    np.array([1000]),
    np.array([1000, 2000]),
    np.array([500,1200,6700,15200]),
])
@pytest.mark.parametrize("frac",[
    1,3,
])
def test_band_filtering(freq,frac):
    """Test freq band data estimation."""

    np.random.shuffle(freq)
    scale = np.random.rand(freq.shape[0])

    signal_split_freqs = pf.signals.sine(frequency=freq, n_samples=441)
    signal_split_freqs.time = (scale*signal_split_freqs.time.T).T

    signal_combined = pf.Signal(data=np.sum(signal_split_freqs.time, axis=0),
                                sampling_rate=signal_split_freqs.sampling_rate)

    band_sig,_ = sp.dsp.band_filter_signal(signal=signal_combined,
                                         freqs=freq,
                                         num_fractions=frac)

    assert band_sig.cshape==signal_split_freqs.cshape

    npt.assert_allclose(np.argmax(np.abs(band_sig.freq),axis=-1),
                        np.argmax(np.abs(signal_split_freqs.freq),axis=-1))


@pytest.mark.parametrize("freq",[
    (3,np.array([1000])),
    (1,np.array([1000, 2000])),
    (1,pf.dsp.filter.fractional_octave_frequencies(num_fractions=1)[0]),
    (3,pf.dsp.filter.fractional_octave_frequencies(num_fractions=3)[0][0:30:5]),
])
def test_closest_freq_band(freq):
    """Test freq band data estimation."""

    np.random.shuffle(freq[1])

    bw, idcs = sp.dsp._closest_frac_octave_data(freqs=freq[1],
                                              num_fractions=freq[0])

    fband_centers,cutoffs = pf.dsp.filter.fractional_octave_frequencies(
        num_fractions=freq[0],
        return_cutoff=True,
        )[1:]

    assert (fband_centers[idcs]>cutoffs[0][idcs]).all()
    assert (fband_centers[idcs]<cutoffs[1][idcs]).all()
    assert (freq[1]>cutoffs[0][idcs]).all()
    assert (freq[1]<cutoffs[1][idcs]).all()

