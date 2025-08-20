"""Useful functions to generate example BRDFs."""
import numpy as np
import os
import pyfar as pf
import sofar as sf

def add_source_receiver_data(sofa, sources, receivers, data):
    """Add source, receiver, and data information to the SOFA file.

    Parameters
    ----------
    sofa : sf.Sofa
        The SOFA file to which the data will be added.
    sources : pf.Coordinates
        The source coordinates, including weights.
    receivers : pf.Coordinates
        The receiver coordinates, including weights.
    data : pf.TimeData or pf.FrequencyData or pf.Signal
        The data to be added to the SOFA file. If the SOFA
        conventions are IR, the data must be a TimeData or Signal
        object. If the SOFA conventions are TF, the data
        must be a FrequencyData object.

    Returns
    -------
    sofa : sf.Sofa
        The SOFA file with the source, receiver, and data
        information added.

    """
    # check inputs
    if not isinstance(sofa, sf.Sofa):
        raise ValueError(
            'Sofa must be a Sofa object.'
        )
    if not isinstance(sources, pf.Coordinates):
        raise ValueError(
            'Sources must be a Coordinates object.'
        )
    if not isinstance(receivers, pf.Coordinates):
        raise ValueError(
            'Receivers must be a Coordinates object.'
        )
    if sources.weights is None:
        raise ValueError(
            'Sources must have weights.'
        )
    if receivers.weights is None:
        raise ValueError(
            'Receivers must have weights.'
        )

    # Source and receiver data
    sofa.EmitterPosition = sources.cartesian
    sofa.EmitterPosition_Units = 'meter'
    sofa.EmitterPosition_Type = 'cartesian'

    sources_sph = sources.spherical_elevation
    sources_sph = pf.rad2deg(sources_sph)
    sofa.SourcePosition = sources_sph
    sofa.SourcePosition_Units = 'degree, degree, metre'
    sofa.SourcePosition_Type = 'spherical'
    if hasattr(sofa, 'SourceWeights'):
        sofa.SourceWeights = sources.weights
    else:
        sofa.add_variable(
            'SourceWeights', sources.weights, 'double', 'E')

    sofa.ReceiverPosition = receivers.cartesian
    sofa.ReceiverPosition_Units = 'meter'
    sofa.ReceiverPosition_Type = 'cartesian'
    if hasattr(sofa, 'ReceiverWeights'):
        sofa.ReceiverWeights = receivers.weights
    else:
        sofa.add_variable(
            'ReceiverWeights', receivers.weights, 'double', 'R')

    if 'IR' in sofa.GLOBAL_SOFAConventions:
        if not isinstance(data, (pf.Signal, pf.TimeData)):
            raise ValueError(
                'Data must be a TimeData or Signal object.'
            )

        sofa.Data_IR = data.time
        sofa.Data_SamplingRate = data.sampling_rate
        sofa.Data_Delay = np.zeros((1, receivers.csize))
    elif 'TF' in sofa.GLOBAL_SOFAConventions:
        if not isinstance(data, (pf.FrequencyData)):
            raise ValueError(
                'Data must be a FrequencyData object.'
            )

        sofa.N = data.frequencies
        sofa.Data_Real = np.real(data.freq)
        sofa.Data_Imag = np.imag(data.freq)
    else:
        conv = sofa.GLOBAL_SOFAConventions
        raise ValueError(
            f'SOFA conventions must contain IR or TF, {conv} not supported.'
        )

    return sofa

def _average_frequencies(data, new_frequencies, domain='pressure'):
    new_shape = np.array(data.freq.shape)
    new_shape[-1] = len(new_frequencies)
    new_data = np.zeros(new_shape)

    for i_freq in range(len(new_frequencies)):
        f_mask = _calculate_f_mask(i_freq, data.frequencies, new_frequencies)
        if domain == 'pressure':
            new_data[..., i_freq] = np.sqrt(
                np.sum(np.abs(data.freq[..., f_mask])**2, -1))
        elif domain == 'energy':
            new_data[...,i_freq] = np.sum(np.abs(data.freq[..., f_mask]),-1)/ \
                                   np.sum(f_mask)
    return pf.FrequencyData(new_data, new_frequencies)

def _calculate_f_mask(i_freq, frequencies_in, frequencies_out):
    if i_freq != 0:
        f_lower = (frequencies_out[i_freq] - frequencies_out[i_freq-1])/2 + \
            frequencies_out[i_freq-1]
    else:
        f_lower = 0
    if i_freq != len(frequencies_out)-1:
        f_upper = (frequencies_out[i_freq+1] - frequencies_out[i_freq])/2 + \
            frequencies_out[i_freq]
    else:
        f_upper = np.inf

    f_mask = (frequencies_in >= f_lower) & (frequencies_in < f_upper)
    return f_mask

def random(
        scattering_coefficients, incident_directions):
    r"""
    Calculate the random-incidence from the directional scattering coefficient.

    Uses the Paris formula [#]_.

    .. math::
        s_{rand} = \sum s(\vartheta,\varphi) \cdot cos(\vartheta) \cdot
        w(\vartheta,\varphi)

    with the scattering coefficients :math:`s(\vartheta,\varphi)`, the area
    weights ``w`` taken from the `incident_directions.weights`,
    and :math:`\vartheta` and :math:`\varphi` are the ``colatitude``
    angle and ``azimuth`` angles from the
    :py:class:`~pyfar.classes.coordinates.Coordinates` object.
    Note that the incident directions should be
    equally distributed to get a valid result. See
    :py:func:`freefield` to calculate the free-field scattering coefficient.

    Parameters
    ----------
    scattering_coefficients : :py:class:`~pyfar.classes.audio.FrequencyData`
        Scattering coefficients for different incident directions. Its cshape
        needs to be (..., incident_directions.csize)
    incident_directions : :py:class:`~pyfar.classes.coordinates.Coordinates`
        Defines the incidence directions of each `scattering_coefficients`
        in a :py:class:`~pyfar.classes.coordinates.Coordinates` object.
        Its cshape needs to match
        the last dimension of `scattering_coefficients`.
        Points contained in `incident_directions` must have the same radii.
        The weights need to reflect the area `incident_directions.weights`.

    Returns
    -------
    random_scattering : :py:class:`~pyfar.classes.audio.FrequencyData`
        The random-incidence scattering coefficient depending on frequency.

    References
    ----------
    .. [#]  H. Kuttruff, Room acoustics, Sixth edition. Boca Raton:
            CRC Press/Taylor & Francis Group, 2017.
    """
    if not isinstance(scattering_coefficients, pf.FrequencyData):
        raise ValueError('coefficients has to be FrequencyData')
    if not isinstance(incident_directions, pf.Coordinates):
        raise ValueError('incident_directions have to be None or Coordinates')
    if incident_directions.cshape[0] != scattering_coefficients.cshape[-1]:
        raise ValueError(
            'the last dimension of coefficients needs be same as '
            'the incident_directions.cshape.')

    theta = incident_directions.colatitude
    weight = np.cos(theta) * incident_directions.weights
    norm = np.sum(weight)
    coefficients_freq = np.swapaxes(scattering_coefficients.freq, -1, -2)
    random_scattering = pf.FrequencyData(
        np.sum(coefficients_freq*weight/norm, axis=-1),
        scattering_coefficients.frequencies
    )
    return random_scattering




