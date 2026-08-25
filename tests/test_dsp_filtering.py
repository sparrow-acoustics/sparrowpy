import numpy as np
import numpy.testing as npt
import pytest
import pyfar as pf
import sparrowpy as sp

@pytest.mark.parametrize("frequencies",[
    np.array([1000]),
    np.array([1000, 2000]),
    np.array([500,1200,6700,15200]),
])
@pytest.mark.parametrize("frac",[
    1,3,
])
@pytest.mark.parametrize("rms",[
    1, (1,1), (2,1,2),
])
def test_band_filtering_multi_dim(frequencies,frac,rms):
    """Test freq band data estimation."""
    noise = pf.signals.noise(n_samples=441, rms=rms)
    band_sig,_ = sp.dsp.band_filter_signal(
        signal=noise, frequencies=frequencies, num_fractions=frac)
    assert band_sig.cshape==(noise.cshape + (len(frequencies),) )


@pytest.mark.parametrize("freq",[
    np.array([1000]),
    np.array([1000, 2000]),
    np.array([500,6700,15200,1200]),
])
@pytest.mark.parametrize("frac",[
    1,3,
])
@pytest.mark.parametrize("n_sigs",[
    1,2,
])
def test_band_filtering(freq,frac,n_sigs):
    """Test freq band data estimation."""

    ff = np.array([freq]*n_sigs)
    scale = (np.ones_like(ff)*
             np.arange(0.1,.5,ff.shape[1])*
             np.arange(0.5,1,ff.shape[0]))
    signal_split_freqs = pf.signals.sine(frequency=ff, n_samples=441)
    signal_split_freqs.time = (scale[...,None]*signal_split_freqs.time)

    signal_combined = pf.Signal(data=np.sum(signal_split_freqs.time, axis=1),
                                sampling_rate=signal_split_freqs.sampling_rate)

    band_sig,_ = sp.dsp.band_filter_signal(signal=signal_combined,
                                           frequencies=freq,
                                           num_fractions=frac)

    assert band_sig.cshape==(signal_split_freqs.cshape)

    npt.assert_allclose(np.argmax(np.abs(band_sig.freq),axis=-1),
                        np.argmax(np.abs(signal_split_freqs.freq),axis=-1))


def test_band_filtering_inputs():
    frequencies = np.array([0,1,2,3])
    with pytest.raises(
            ValueError,
            match="Input frequencies must be greater than zero."):
        sp.dsp.band_filter_signal(frequencies=frequencies,
                                signal=pf.signals.noise(n_samples=400),
                                num_fractions=1)


    frequencies = np.array([10,20000,-3])
    with pytest.raises(
            ValueError,
            match="Input frequencies must be greater than zero."):
        sp.dsp.band_filter_signal(frequencies=frequencies,
                                signal=pf.signals.noise(n_samples=400),
                                num_fractions=1)


    with pytest.raises(
            ValueError,
            match="Number of octave fractions must be greater than zero."):
        sp.dsp.band_filter_signal(frequencies=np.array([1,2,3]),
                                signal=pf.signals.noise(n_samples=400),
                                num_fractions=0)

    with pytest.raises(
            ValueError,
            match="Number of octave fractions must be greater than zero."):
        sp.dsp.band_filter_signal(frequencies=np.array([1,2,3]),
                            signal=pf.signals.noise(n_samples=400),
                            num_fractions=-5)
