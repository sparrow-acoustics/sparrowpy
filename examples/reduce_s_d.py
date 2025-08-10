# %%
import pyfar as pf
import sofar as sf
import numpy as np

from tqdm import tqdm
# %%

def average_frequencies(data, new_frequencies, domain='pressure'):
    new_shape = np.array(data.freq.shape)
    new_shape[-1] = len(new_frequencies)
    new_data = np.zeros(new_shape)

    for i_freq in range(len(new_frequencies)):
        f_mask = _calculate_f_mask(i_freq, data.frequencies, new_frequencies)
        if domain == 'pressure':
            new_data[..., i_freq] = np.sqrt(
                np.sum(np.abs(data.freq[..., f_mask])**2, -1))
        elif domain == 'energy':
            new_data[..., i_freq] = np.sum(np.abs(data.freq[..., f_mask]), -1)/np.sum(f_mask)
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


def get_s_rand_from_bsc(file_in = 'examples/resources/triangle_sim_optimal.s_d.sofa', freq_out=None):
    sofa = sf.read_sofa(file_in)
    bsc, sources, receivers = pf.io.convert_sofa(sofa)
    sources.weights = sofa.SourceWeights
    receivers.weights = sofa.ReceiverWeights

    # apply scaling factor
    bsc._frequencies /= 8

    # calculate scattering coefficient from BSC
    spec_direction = sources.copy()
    spec_direction.azimuth += np.pi
    i_spec_dir = sources.find_nearest(spec_direction)[0][0]
    scattering = bsc[np.arange(len(i_spec_dir)), i_spec_dir]

    # average scattering coefficient to random incident
    scattering_rand = random(scattering, sources)

    if freq_out is None:
        frequencies_out = pf.dsp.filter.fractional_octave_frequencies(
            1, (np.min(bsc.frequencies/8), np.max(bsc.frequencies/8)),
        )[1]
    else:
        frequencies_out=freq_out

    scattering_rand_oct = average_frequencies(
        scattering_rand, frequencies_out, domain='energy')

    return scattering_rand_oct, sources, receivers


def get_bsc(file_in = 'examples/resources/triangle_sim_optimal.s_d.sofa', freq_out=None):
    sofa = sf.read_sofa(file_in)
    bsc, sources, receivers = pf.io.convert_sofa(sofa)
    sources.weights = sofa.SourceWeights
    receivers.weights = sofa.ReceiverWeights

    # mirror missing data
    bsc_incident_directions = receivers.copy()
    bsc_scattering_directions = receivers.copy()
    bsc_mirrored = pf.FrequencyData(
        np.zeros((bsc_incident_directions.csize,
                  bsc_scattering_directions.csize,
                  bsc.n_bins)),
        bsc.frequencies/8)
    for i_source in tqdm(range(receivers.csize)):
        # mirror results from previous directions for Azimuth > 90°
        if bsc_incident_directions[i_source].azimuth > np.pi/2:
            az_is = bsc_incident_directions[i_source].azimuth[0]
            if az_is < np.pi:  # 180:
                az_mirror = np.pi - az_is
            elif az_is < 3/2*np.pi:  # 270°
                az_mirror = np.abs(np.pi - az_is)
            else:
                az_mirror = 2*np.pi - az_is

            delta_azimuth = az_is - az_mirror

            # find the correct incident direction due to symmetry
            find_incident = pf.Coordinates.from_spherical_elevation(
                az_mirror,
                bsc_incident_directions.elevation[i_source],
                bsc_incident_directions.radius[i_source])
            i_source_mirror = bsc_incident_directions.find_nearest(
                find_incident, distance_measure='spherical_radians',
                radius_tol=1e-13)[0][0]

            # rotate the scattering data to the correct azimuth
            shifted_coords = bsc_scattering_directions.copy()
            shifted_coords.azimuth -= delta_azimuth
            idx_scattering = bsc_incident_directions.find_nearest(
                shifted_coords, distance_measure='spherical_radians',
                radius_tol=1e-13)[0]
            bsc_mirrored.freq[i_source, :, :] = bsc.freq[i_source_mirror,
                                                         idx_scattering, :]

    if freq_out is None:
        frequencies_out = pf.dsp.filter.fractional_octave_frequencies(
            1, (np.min(bsc.frequencies/8), np.max(bsc.frequencies/8)),
        )[1]
    else:
        frequencies_out=freq_out

    bsc_octave = average_frequencies(
        bsc_mirrored, frequencies_out, domain='energy')

    for k in range(bsc_octave.freq.shape[1]):
        if np.sum(bsc_octave.freq[:,k])!=0:
            bsc_octave.freq[:,k] /= np.sum(bsc_octave.freq[:,k])

    return bsc_octave, receivers, receivers


if __name__=="__main__":
    get_bsc()
