"""Contain functions to generate IRs from sparrowpy ETCs."""
import numpy as np
import pyfar as pf


def energy_time_curve_from_impulse_response(
        signal, delta_time=0.01, bandwidth=None):
    r"""Calculate the energy time curve from impulse responses.

    .. math::
        E_n = \sum_{g(k)+1}^{g(k-1)} h(i)^2 \cdot \frac{f_s/2}{BW}

    where :math:`g(k)=floor(k \cdot f_s \cdot \Delta t)` representing the
    range of each energy window with length :math:`\Delta t` and
    :math:`h(i)` is the impulse response signal. :math:`BW` is the bandwidth
    of the energy time curve, which is set to half the sampling rate
    by default.


    Parameters
    ----------
    signal : pyfar.Signal
        The impulse responses from which the energy time curve is
        calculated. The cshape should be ``(n_bands, ...)``.
    delta_time : float, optional
        The time resolution of the energy time curve,
        by default ``0.01`` seconds.
    bandwidth : float, array_like, optional
        Bandwidth for each signal channel, by default None, which means its
        assumed for the full bandwidth :math:`BW = f_s/2`.
        The shape should be ``(n_bands)``.

    Returns
    -------
    etc : :py:class:`pyfar.TimeData`
        The energy time curve of the impulse responses.

    Examples
    --------
    Generate the energy time curve from an impulse for full bandwidth.

    .. plot::

        >>> import pyfar as pf
        >>> import sparrowpy as sp
        >>> n_samples = 44100
        >>> impulse_response = pf.signals.impulse(n_samples)
        >>> etc = sp.dsp.energy_time_curve_from_impulse_response(
        ...     impulse_response, delta_time=0.01)
        >>> ax = pf.plot.time(impulse_response, dB=True)
        >>> ax = pf.plot.time(etc, ax=ax, dB=True, log_prefix=10)
        >>> ax.set_title("Energy time curve from impulse response")

    Generate the energy time curve from octave band filtered white noise.

    .. plot::

        >>> import pyfar as pf
        >>> import sparrowpy as sp
        >>> n_samples = 44100
        >>> white_noise = pf.signals.noise(n_samples, 'white')
        >>> cutoff_freq = pf.dsp.filter.fractional_octave_frequencies(
        ...     1, (1e3, 20e3), return_cutoff=True)[2]
        >>> bw = cutoff_freq[1] - cutoff_freq[0]
        >>> filtered_white_noise = pf.dsp.filter.fractional_octave_bands(
        ...     white_noise, 1, frequency_range=(1e3, 20e3),)
        >>> etc = sp.dsp.energy_time_curve_from_impulse_response(
        ...     filtered_white_noise, 0.01, bw)
        >>> ax = pf.plot.time(etc, dB=True, log_prefix=10)
        >>> ax.set_title("ETC from octave band filtered white noise")

    """
    if not isinstance(signal, pf.Signal):
        raise TypeError("signal must be a pyfar Signal object.")

    if not isinstance(delta_time, (float, int)):
        raise TypeError("delta_time must be a float or int.")
    if delta_time <= 0:
        raise ValueError("delta_time must be positive.")

    if bandwidth is not None:
        if isinstance(bandwidth, (float, int)):
            if bandwidth <= 0:
                raise ValueError("bandwidth must be positive.")
        else:
            bandwidth = np.asarray(bandwidth)
            if np.any(bandwidth <= 0):
                raise ValueError("All bandwidth values must be positive.")
            if bandwidth.shape != signal.cshape[:1]:
                raise ValueError(
                    f"bandwidth shape {bandwidth.shape} does not "
                    f"match signal bands {signal.cshape[:1]}",
                )

    if bandwidth is None:
        bandwidth = signal.sampling_rate / 2
    n_samples_E = int(np.ceil(
        signal.n_samples / signal.sampling_rate / delta_time))

    g_k = np.asarray(
        np.arange(n_samples_E)*signal.sampling_rate*delta_time, dtype=int)
    etc = pf.TimeData(
        np.zeros((*signal.cshape, n_samples_E)),
        np.arange(n_samples_E) * delta_time,
        )
    bw_factor = np.asarray((signal.sampling_rate/2)/bandwidth)
    for _ in range(signal.cdim-1):
        bw_factor = bw_factor[..., np.newaxis]

    for k in range(n_samples_E):
        upper = g_k[k+1] if k < n_samples_E-1 else signal.n_samples
        lower = g_k[k]
        etc.time[..., k] = np.sum(
            signal.time[..., lower:upper]**2, axis=-1) * bw_factor

    return etc
