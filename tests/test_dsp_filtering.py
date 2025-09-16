import pytest
import numpy as np
import numpy.testing as npt
import sparrowpy as sp
import pyfar as pf

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
                                         frequencies=freq,
                                         num_fractions=frac,
                                         )

    assert band_sig.cshape==signal_split_freqs.cshape

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


@pytest.mark.parametrize("freq",[
    (3,np.array([1000])),
    (1,np.array([1000, 2000])),
    (2,pf.dsp.filter.fractional_octave_frequencies(num_fractions=1)[0]),
    (8,pf.dsp.filter.fractional_octave_frequencies(num_fractions=3)[0][0:30:5]),
])
def test_closest_freq_band(freq):
    """Test freq band data estimation."""

    np.random.shuffle(freq[1])

    bw, idcs = sp.dsp._closest_frac_octave_data(frequencies=freq[1],
                                              num_fractions=freq[0])

    fband_centers,cutoffs = pf.dsp.filter.fractional_octave_frequencies(
        num_fractions=freq[0],
        return_cutoff=True,
        )[1:]

    assert (fband_centers[idcs]>cutoffs[0][idcs]).all()
    assert (fband_centers[idcs]<cutoffs[1][idcs]).all()
    assert (freq[1]>cutoffs[0][idcs]).all()
    assert (freq[1]<cutoffs[1][idcs]).all()

def test_closest_freq_band_inputs():
    frequencies = np.array([0,1,2,3])
    with pytest.raises(
            ValueError,
            match="Input frequencies must be greater than zero."):
        sp.dsp._closest_frac_octave_data(frequencies=frequencies,
                                        num_fractions=1)

    frequencies = np.array([10,20000,-3])
    with pytest.raises(
            ValueError,
            match="Input frequencies must be greater than zero."):
        sp.dsp._closest_frac_octave_data(frequencies=frequencies,
                                        num_fractions=1)

    frequencies=np.array([100,200,300])
    with pytest.raises(
            ValueError,
            match="Number of octave fractions must be greater than zero."):
        sp.dsp._closest_frac_octave_data(frequencies=frequencies,
                                        num_fractions=0)

    with pytest.raises(
            ValueError,
            match="Number of octave fractions must be greater than zero."):
        sp.dsp._closest_frac_octave_data(frequencies=frequencies,
                                        num_fractions=-5)

    frequencies=np.array([1000,1001])
    with pytest.warns(
          match="Multiple input frequencies in the same freq. band.\n" +
                "You may want to revise your input frequencies or " +
                "increase the filter bandwidths.",
                ):
        sp.dsp._closest_frac_octave_data(frequencies=frequencies,
                                        num_fractions=1)
