"""Contain functions to generate IRs from sparrowpy ETCs."""
import numpy as np
import sofar as sf
import pyfar as pf
import sparrowpy

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

    .. math:: \mu = \min{\left(\frac{4 \pi c^3 \cdot t^2}{V}, \mu_{max}\right)}

    with :math:`t` being the time vector in seconds based on ``sampling_rate``
    and ``n_samples`` and :math:`\mu_{max}` being the
    maximum reflection density.

    .. note::
        This function can be used to generate the Dirac sequence for the
        room impulse response synthesis using
        :py:func:`pyfar.signals.dirac_sequence`.

    Parameters
    ----------
    room_volume : float
        Volume of the room :math:`V` in :math:`m^3`.
    n_samples : int
        The length of the signal in samples.
    speed_of_sound : float, None, optional
        Speed of sound in the room.
        By default, the :py:attr:`~pyfar.constants.reference_speed_of_sound`
        is used.
    max_reflection_density : int, optional
        The maximum reflection density. The default is None, which means
        that the reflection density is not limited.
    sampling_rate : int, optional
        The sampling rate in Hz. The default is ``44100``.

    Returns
    -------
    reflection_density : pyfar.TimeData
        reflection density :math:`\mu` in :math:`1/s^2` over time.
    t_start : float
        The dirac sequence generation starts after :math:`t_start`.

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
    and :math:`\mu` being the ``reflection_density`` in a room.
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
        The default is 44100.
    seed : int, None, optional
        The seed for the random generator. Pass a seed to obtain identical
        results for multiple calls. The default is ``None``, which will yield
        different results with every call.
        See :py:func:`numpy.random.default_rng` for more information.

    Returns
    -------
    dirac_sequence : pyfar.Signal
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

        >>> import sparrowpy as sp
        >>> import pyfar as pf
        >>> n_samples = 22050
        >>> reflection_density, t_0 = sp.dsp.reflection_density_room(
        ...     5000, n_samples)
        >>> dirac_sequence = sp.dsp.dirac_sequence(
        ...     reflection_density, n_samples, t_start=t_0)
        >>> ax = pf.plot.time(dirac_sequence, linewidth=.5)
        >>> ax.set_title("Dirac sequence")

    Generate a Dirac sequence based on a custom reflection density.

    .. plot::

        >>> import sparrowpy as sp
        >>> import pyfar as pf
        >>> import numpy as np
        >>> n_samples = 22050
        >>> reflection_density = pyfar.TimeData(
        ...     np.ones(n_samples)*10000, np.arange(n_samples)/44100)
        >>> dirac_sequence = sp.dsp.dirac_sequence(
        ...     reflection_density, n_samples, t_start=0)
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