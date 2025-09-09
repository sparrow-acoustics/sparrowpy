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

        >>> import pyfar as pf
        >>> n_samples = 8800
        >>> sampling_rate = 44100
        >>> for v in [100, 500, 1000, 5000]:
        >>>     reflection_density, t_0 = pf.signals.reflection_density_room(
        ...         v, n_samples, sampling_rate)
        >>>     ax = pf.plot.time(
        ...         reflection_density,
        ...         label=f"V={v} m$^3$ -> $t_0={t_0*1e3:.1f}$ ms")
        >>> ax.legend()
        >>> ax.set_title("Reflection density")

    """
    if speed_of_sound is None:
        speed_of_sound = pf.constants.reference_speed_of_sound

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