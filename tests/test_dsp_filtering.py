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
                                         num_fractions=frac,
                                         )

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


def test_closest_frac_octave_data_bandwidth():
    num_fractions = 3
    frequency_range = (25, 12e3)
    _, fband_centers, cutoffs = pf.dsp.filter.fractional_octave_frequencies(
        num_fractions=num_fractions,
        frequency_range=frequency_range,
        return_cutoff=True,
        )
    bw, _ = sp.dsp._closest_frac_octave_data(
        frequencies=fband_centers, num_fractions=num_fractions)
    npt.assert_allclose(bw, cutoffs[1]-cutoffs[0])


@pytest.mark.parametrize("freq",[
    (3,np.array([1000])),
    (1,np.array([1000, 2000])),
    (2,pf.dsp.filter.fractional_octave_frequencies(num_fractions=1)[0]),
    (8,pf.dsp.filter.fractional_octave_frequencies(num_fractions=3)[0][0:30:5]),
])
def test_closest_freq_band_idcs(freq):
    """Test freq band data estimation."""

    _, idcs = sp.dsp._closest_frac_octave_data(frequencies=freq[1],
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
