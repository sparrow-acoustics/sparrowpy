
"""Module for filter generation and signal processing in sparrowpy."""
import numpy as np
import pyfar as pf

def reflection_density_room(
        room_volume, n_samples, speed_of_sound=None,
        max_reflection_density=None, sampling_rate=44100):
    r"""
    Calculates the reflection density and starting time in a diffuse room.

    The reflection density and starting time based on the chapter
    5.3.4 of [#]_.
    The starting time :math:`t_\text{start}` can be calculated based on:

    .. math::
        t_\text{start} = \left(\frac{2 V \cdot \ln(2)}{4 \pi c^3}\right)^{1/3}

    where :math:`V` is the room volume in :math:`m^3` and :math:`c` is the
    speed of sound in the room. The reflection density :math:`\mu`
    is calculated based on the following equation:

    .. math::
        \mu = \min{\left(\frac{4 \pi c^3 \cdot t^2}{V} , \mu_{max}\right)}

    with :math:`t` being the time vector in seconds based on ``sampling_rate``
    and ``n_samples`` and :math:`\mu_{max}` being the
    maximum reflection density.

    .. note::
        This function can be used to generate the Dirac sequence for the
        room impulse response synthesis using
        :py:func:`sparrowpy.dsp.dirac_sequence`.

    Parameters
    ----------
    room_volume : float
        Volume of the room :math:`V` in :math:`m^3`.
    n_samples : int
        The length of the signal in samples.
    speed_of_sound : float, None, optional
        Speed of sound in the room.
        By default, 343.2 m/s is used.
    max_reflection_density : int, optional
        The maximum reflection density. The default is None, which means
        that the reflection density is not limited.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is ``44100``.

    Returns
    -------
    reflection_density : :py:class:`pyfar.TimeData`
        reflection density :math:`\mu` in :math:`1/s^2` over time.
    t_start : float
        The dirac sequence generation starts after :math:`t_\text{start}`.

    References
    ----------
    .. [#] D. Schröder, “Physically based real-time auralization of
           interactive virtual environments,” PhD Thesis, Logos-Verlag,
           Berlin, 2011. [Online].
           Available: https://publications.rwth-aachen.de/record/50580

    Examples
    --------
    Generate the reflection densities for a room with a different volumes.

    .. plot::

        >>> import sparrowpy as sp
        >>> import pyfar as pf
        >>> n_samples = 8800
        >>> sampling_rate = 44100
        >>> for v in [100, 500, 1000, 5000]:
        >>>     reflection_density, t_0 = sp.dsp.reflection_density_room(
        ...         v, n_samples, sampling_rate)
        >>>     ax = pf.plot.time(
        ...         reflection_density,
        ...         label=f"V={v} m$^3$ -> $t_0={t_0*1e3:.1f}$ ms")
        >>> ax.legend()
        >>> ax.set_title("Reflection density")
    """

    if speed_of_sound is None:
        speed_of_sound = 343.2

    # check input
    room_volume = float(room_volume)
    n_samples = int(n_samples)
    if max_reflection_density is not None:
        max_reflection_density = int(max_reflection_density)
        if max_reflection_density <= 0:
            raise ValueError("max_reflection_density must be positive.")

    speed_of_sound = float(speed_of_sound)
    if speed_of_sound <= 0:
        raise ValueError("speed_of_sound must be positive.")
    if room_volume <= 0:
        raise ValueError("room_volume must be positive.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    times = pf.Signal(np.zeros(n_samples), sampling_rate).times
    # calculate the reflection density
    mu = 4 * np.pi * speed_of_sound**3 * times**2 / room_volume
    if max_reflection_density is not None:
        mu[mu > max_reflection_density] = max_reflection_density
    mu = pf.TimeData(mu, times)

    # calculate the time of the first reflection
    t_start = (2*room_volume*np.log(2)/(4*np.pi*speed_of_sound**3))**(1/3)

    # return the reflection density and the starting time
    return mu, t_start

def dirac_sequence(
        reflection_density, n_samples, t_start=0, sampling_rate=44100,
        seed=None):
    r"""Dirac sequence based on the reflection density over time.

    The Dirac sequence is generated based on the chapter 5.3.4 of [#]_.

    The time difference between each dirac in the sequence is Poisson
    distributed and can be calculated based on:

    .. math:: \Delta t_a = \frac{1}{\mu} \cdot \ln{\frac{1}{z}}

    with z being a random number in the range of :math:`z \in (0, 1]`
    and :math:`\mu` being the ``reflection_density`` over time.
    Each dirac has an amplitude of 1 or -1, which is chosen
    randomly with equal probability.
    The dirac sequence generation starts after :math:`t_\text{start}`.

    Parameters
    ----------
    reflection_density : pyfar.TimeData
        reflection density :math:`\mu` in :math:`1/s^2` over time.
        An error is raised if the reflection sensitivity is greater than
        sampling_rate/2. Schröder suggested a maximum reflection density of
        sampling_rate/4 :math:`1/s^2`.
    n_samples : int
        The length of the dirac sequence in samples.
    t_start : float
        The dirac sequence generation starts after :math:`t_\text{start}`
        in seconds. The default is ``0``.
    sampling_rate : int, optional
        The sampling rate of the dirac sequence in Hz.
        The default is 44100 Hz.
    seed : int, None, optional
        The seed for the random generator. Pass a seed to obtain identical
        results for multiple calls. The default is ``None``, which will yield
        different results with every call.
        See :py:func:`numpy.random.default_rng` for more information.

    Returns
    -------
    dirac_sequence : :py:class:`pyfar.Signal`
        Signal of the generated dirac impulse sequence.

    References
    ----------
    .. [#] D. Schröder, “Physically based real-time auralization of
           interactive virtual environments,” PhD Thesis, Logos-Verlag,
           Berlin, 2011. [Online].
           Available: https://publications.rwth-aachen.de/record/50580

    Examples
    --------
    Generate a Dirac sequence based on the reflection density of a room
    with a volume of 5000 m³.

    .. plot::

        >>> import pyfar as pf
        >>> import sparrowpy as sp
        >>> n_samples = 22050
        >>> reflection_density, t_0 = sp.dsp.reflection_density_room(
        ...     5000, n_samples, max_reflection_density=5e3)
        >>> dirac_sequence = sp.dsp.dirac_sequence(
        ...     reflection_density, n_samples, t_start=t_0, seed=0)
        >>> ax = pf.plot.time(dirac_sequence, linewidth=.5)
        >>> ax.set_title("Dirac sequence")

    Generate a Dirac sequence based on a constant reflection density.

    .. plot::

        >>> import pyfar as pf
        >>> import numpy as np
        >>> import sparrowpy as sp
        >>> n_samples = 22050
        >>> reflection_density = pf.TimeData(
        ...     np.ones(n_samples)*100, np.arange(n_samples)/44100)
        >>> dirac_sequence = sp.dsp.dirac_sequence(
        ...     reflection_density, n_samples, t_start=0, seed=0)
        >>> ax = pf.plot.time(dirac_sequence, linewidth=.5)
        >>> ax.set_title("Dirac sequence")
    """
    # check input
    if not isinstance(reflection_density, pf.TimeData):
        raise ValueError(
            "reflection_density must be a pyfar.TimeData object.")
    if t_start < 0:
        raise ValueError("t_start must be positive.")
    if np.any(reflection_density.time > sampling_rate / 2):
        raise ValueError(
            "The reflection density must be less than sampling_rate/2.")

    rng = np.random.default_rng(seed)
    dirac_sequence = pf.Signal(np.zeros(n_samples), sampling_rate)
    mu_times = reflection_density.times
    t_current = t_start
    i_current = np.argmin(np.abs(t_current-mu_times))
    t_max = reflection_density.times[-1]
    while True:
        # calculate next event time
        z = -rng.uniform(-1, 0) # uniform distribution in (0, 1]
        # Equation (5.43) interval size
        delta_ta = 1 / reflection_density.time[..., i_current] * np.log(1 / z)
        t_current += delta_ta

        if t_current > t_max:
            break

        i_current = np.argmin(np.abs(t_current-mu_times))

        dirac_sequence.time[..., i_current] = rng.choice([-1, 1], p=[0.5, 0.5])

    return dirac_sequence

def weight_signal_by_etc(
    energy_time_curve: pf.TimeData,
    signal: pf.Signal,
    bandwidth=None,
) -> pf.Signal:
    r"""
    Weight a signal with a given energy time curve.

    The signals are weighed by the respective energy time curve
    over time after Chapter 5.3.4 of [#]_:

    .. math::
        h_i = \nu_i \cdot \sqrt{\frac{E_n(k)}{\sum^{g(k)}_{g(k-1)+1} \nu_i^2}}
        \cdot \sqrt{\frac{BW}{f_s/2}}

    where :math:`h_i` and :math:`\nu_i` represent respectively the
    weighted output signal and the input signal at a given time sample
    :math:`i`. :math:`g(k)=\lfloor k \cdot f_s \cdot \Delta t \rfloor`
    represents the range of each energy window with given length
    :math:`\Delta t` of the ETC entry :math:`E(k)` with index :math:`k`.
    :math:`BW` is the bandwidth
    of the energy time curve.

    Parameters
    ----------
    energy_time_curve: :py:class:`pyfar.TimeData`
        Energy time curve of a sound propagation simulation of cshape
        ``(..., n_freq_bands)`` and broadcastable to ``signal``.
        The ETC entries must be equally spaced in time.
    signal: :py:class:`pyfar.Signal`
        signal to be weighted by the etc of cshape
        ``(..., n_freq_bands)`` and broadcastable to energy_time_curve.
    bandwidth: np.ndarray
        Bandwidth for the frequency band in Hz of shape
        ``(n_freq_bands)``. By default, signal will be processed
        as full spectrum ``sampling_rate/2``.

    Returns
    -------
    weighted_signal : :py:class:`pyfar.Signal`
        signal weighted by the energy_time_curve.
        The cshape matches the cshape of the etc.

    References
    ----------
    .. [#] D. Schröder, “Physically based real-time auralization of
           interactive virtual environments,” PhD Thesis, Logos-Verlag,
           Berlin, 2011. [Online].
           Available: https://publications.rwth-aachen.de/record/50580

    Examples
    --------
    Weight white noise by a single broadband exponential decay etc.

    .. plot::

        >>> import pyfar as pf
        >>> import sparrowpy as sp
        >>> import numpy as np
        >>> n_samples = 44100
        >>> white_noise = pf.dsp.normalize(pf.signals.noise(n_samples,rms=1))
        >>> delta_t = 1/1000
        >>> times = np.arange(0,white_noise.times[-1],delta_t)
        >>> decay = np.exp(-4*times)
        >>> etc = pf.TimeData(data=decay,times=times)
        >>> weighted_noise = sp.dsp.weight_signal_by_etc(energy_time_curve=etc,
        ...                                               signal=white_noise)
        >>> ax=pf.plot.time(white_noise,label="input signal",dB=True)
        >>> ax=pf.plot.time(weighted_noise,label="weighted signal",
        ...                 ax=ax,dB=True)
        >>> ax.set_title("Signal weighting by exponential decaying ETC")
        >>> ax.legend()


    Weight white noise channels with varying bandwidths by a constant ETC.

    .. plot::

        >>> import pyfar as pf
        >>> import sparrowpy as sp
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> n_samples = 44100
        >>> bandwidth = np.array([2000,1000,500])
        >>> white_noise = pf.dsp.normalize(pf.signals.noise(n_samples,rms=1))
        >>> white_noise.time = np.repeat(white_noise.time,bandwidth.shape[0],
        ...                              axis=0)
        >>> delta_t = 1/1000
        >>> times = np.arange(0,white_noise.times[-1],delta_t)
        >>> etc =pf.TimeData(data=np.ones((bandwidth.shape[0],times.shape[0])),
        ...                  times=times)
        >>> weighted_noise_bandwise = sp.dsp.weight_signal_by_etc(
        ...     energy_time_curve=etc,
        ...     signal=white_noise,
        ...     bandwidth=bandwidth,
        ... )
        >>> ax=pf.plot.time(weighted_noise_bandwise,
        ...              label=[f"{bandwidth[i]}Hz bandwidth" \
        ...                             for i in range(bandwidth.shape[0])],
        ...              dB=True)
        >>> ax.legend()
        >>> ax.set_title("Bandwidth-scaled white noise")

    Weight white noise of equal bandwidth by varied exponential decay ETCs.

    .. plot::

        >>> import pyfar as pf
        >>> import sparrowpy as sp
        >>> import numpy as np
        >>> n_samples = 44100
        >>> n_channels = 5
        >>> white_noise = pf.signals.noise(
        ...                     n_samples,
        ...                     rms=np.ones((n_channels,)),
        ...                     )
        >>> delta_t = 1/1000
        >>> times = np.arange(0,white_noise.times[-1],delta_t)
        >>> decay = np.empty((n_channels,times.shape[0]))
        >>> for i in range(n_channels):
        ...     decay[i,:] = np.exp(-3*i*times)
        >>> etc = pf.TimeData(data=decay,times=times)
        >>> weighted_noise_bandwise = sp.dsp.weight_signal_by_etc(
        ...     energy_time_curve=etc,
        ...     signal=white_noise,
        ...     bandwidth=200*np.ones((n_channels,)),
        ... )
        >>> ax=pf.plot.time(
        ...     weighted_noise_bandwise,
        ...     label=[f"exp(-{i*3}t) decay" for i in range(n_channels)],
        ...     )
        >>> ax.legend()
        >>> ax.set_title(
        ...     "Multiple white noise channels weighted by independent ETCs"
        ...     )

    """
    if bandwidth is None:
        bandwidth = signal.sampling_rate / 2

    if isinstance(bandwidth, (float, int)):
        if bandwidth <= 0:
            raise ValueError("Bandwidth must be positive.")
    else:
        bandwidth = np.asarray(bandwidth)
        if np.any(bandwidth <= 0):
            raise ValueError("All bandwidth values must be positive.")
        if bandwidth.shape != signal.cshape[(-bandwidth.ndim):]:
            raise ValueError(
                f"bandwidth shape {bandwidth.shape} does not "
                "match signal bands "
                f"{signal.cshape[-bandwidth.ndim:]}",
            )
        if bandwidth.shape != energy_time_curve.cshape[(-bandwidth.ndim):]:
            raise ValueError(
                f"bandwidth shape {bandwidth.shape} does not "
                "match etc shape "
                f"{energy_time_curve.cshape[-bandwidth.ndim:]}",
            )

    if type(energy_time_curve) is not pf.TimeData:
        raise ValueError("ETC must be a pyfar.TimeData object.")

    if type(signal) is not pf.Signal:
        raise ValueError("Input signal must be a pyfar.Signal object.")

    if not (np.abs(energy_time_curve.times[1:]-energy_time_curve.times[:-1] -
        energy_time_curve.times[1]-energy_time_curve.times[0]) < 1e-12 ).all():
        raise ValueError("ETC entries must be equally spaced in time.")

    rs_factor = signal.sampling_rate*(energy_time_curve.times[1] -
                                      energy_time_curve.times[0])

    weighted_signal_arr = np.zeros(energy_time_curve.cshape +
                              (signal.n_samples,))

    for sample_i in range(energy_time_curve.n_samples):
        lower = int(sample_i * rs_factor)
        upper = min(int((sample_i+1) * rs_factor),signal.n_samples)

        signal_sec = signal.time[...,lower:upper]
        div = np.sum(signal_sec**2,axis=-1)

        scale = np.divide(energy_time_curve.time[...,sample_i]*
                                                (upper-lower)/rs_factor,
                          div,
                          out=np.zeros_like(energy_time_curve.time[...,sample_i]),
                          where=div!=0)

        etc_weight = np.sqrt(scale) * np.sqrt(bandwidth /
                                              (signal.sampling_rate/2))

        weighted_signal_arr[...,lower:upper]=(
            etc_weight[...,None]*signal_sec
        )

    weighted_signal = pf.Signal(weighted_signal_arr, signal.sampling_rate)

    return weighted_signal

def energy_time_curve_from_impulse_response(
        signal, delta_time=0.01, bandwidth=None):
    r"""Calculate the energy time curve from impulse responses.

    .. math::
        E_n = \sum_{g(k)}^{g(k+1)} h(i)^2 \cdot \frac{f_s/2}{BW}

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
    bw_factor = bw_factor.reshape(bw_factor.shape + (1,) * (signal.cdim - 1))

    for k in range(n_samples_E):
        upper = g_k[k+1] if k < n_samples_E-1 else signal.n_samples
        lower = g_k[k]
        etc.time[..., k] = np.sum(
            signal.time[..., lower:upper]**2, axis=-1) * bw_factor

    return etc
