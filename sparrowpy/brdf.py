"""Contain functions to create BRDFs from scattering coefficients."""
import numpy as np
import sofar as sf
import pyfar as pf
import sparrowpy


def create_from_scattering(
        source_directions,
        receiver_directions,
        scattering_coefficient,
        absorption_coefficient=None,
        file_path=None,
        ):
    r"""Create the BRDF from a scattering coefficient and write to SOFA file.

    The scattering coefficient is assumed to be anisotropic and based on [#]_.
    The BRDF is discretized as follows:

    .. math::
        \rho(\mathbf{\Omega_i}, \mathbf{\Omega_o}) =
        \frac{(1-s)(1-\alpha)}{\mathbf{\Omega_i} \cdot \mathbf{n}}
        \frac{1}{w_o} \delta(\mathbf{\Omega_i}-M(\mathbf{\Omega_o})) +
        \frac{s(1-\alpha)}{\pi}

    where:
        - :math:`\mathbf{\Omega_i}` and :math:`\mathbf{\Omega_o}` are the
          incident and outgoing directions, respectively.
        - :math:`s` is the scattering coefficient.
        - :math:`\alpha` is the absorption coefficient.
        - :math:`\mathbf{n}` is the normal vector of the surface.
        - :math:`\delta` is the Dirac delta function.
        - :math:`w_o` is weighting factor of the outgoing angular sector
          (unit sphere).
        - :math:`M` s the mirror reflection transformation
          :math:`M(\theta, \phi)=M(\theta, \pi-\phi)`.

    Note that the weights doesn't need to be normalized,
    they get scaled as required.

    Parameters
    ----------
    source_directions : :py:class:`~pyfar.classes.coordinates.Coordinates`
        source directions for the BRDF, should contain weights.
        cshape of data should be (n_sources)
    receiver_directions : :py:class:`~pyfar.classes.coordinates.Coordinates`
        receiver directions for the BRDF, should contain weights.
        cshape of data should be (n_receivers)
    scattering_coefficient : :py:class:`~pyfar.classes.audio.FrequencyData`
        frequency dependent scattering coefficient data.
        cshape of data should be (1, ).
    absorption_coefficient : :py:class:`~pyfar.classes.audio.FrequencyData`
        frequency dependent absorption coefficient data, by default
        no absorption.
        cshape of data should be (1, ).
    file_path : string, path
        path where sofa file should be saved, by default no file is saved.

    Returns
    -------
    brdf : :py:class:`~pyfar.classes.audio.FrequencyData`
        BRDF data.

    References
    ----------
    .. [#]  S. Siltanen, T. Lokki, S. Kiminki, and L. Savioja, “The room
            acoustic rendering equation,” The Journal of the Acoustical
            Society of America, vol. 122, no. 3, pp. 1624-1635, 2007.

    Examples
    --------
    >>> import pyfar as pf
    >>> import sparrowpy as sp
    >>> import numpy as np
    >>> scattering_coefficient = pf.FrequencyData(0.5, [100])
    >>> directions = pf.samplings.sph_gaussian(sh_order=3)
    >>> directions = directions[directions.z > 0]
    >>> brdf = sp.brdf.create_from_scattering(
    ...     directions, directions,
    ...     scattering_coefficient)

    """
    if (
            not isinstance(scattering_coefficient, pf.FrequencyData) or
            not scattering_coefficient.cshape == (1,)):
        raise TypeError(
            'scattering_coefficient must be a pf.FrequencyData object'
            'with shape (1,)')
    if not isinstance(source_directions, pf.Coordinates):
        raise TypeError(
            'source_directions must be a pf.Coordinates object')
    if not isinstance(receiver_directions, pf.Coordinates):
        raise TypeError(
            'receiver_directions must be a pf.Coordinates object')
    if absorption_coefficient is None:
        absorption_coefficient = pf.FrequencyData(
            np.zeros_like(scattering_coefficient.frequencies),
            scattering_coefficient.frequencies)

    brdf = np.zeros((
        source_directions.csize, receiver_directions.csize,
        scattering_coefficient.n_bins))

    receiver_weights = receiver_directions.weights
    receiver_weights *= 2 * np.pi / np.sum(receiver_weights)
    scattering_flattened = np.real(scattering_coefficient.freq.flatten())
    image_source = source_directions.copy()
    image_source.azimuth += np.pi
    i_receiver = receiver_directions.find_nearest(image_source)[0][0]
    cos_factor = (np.cos(
            source_directions.colatitude[
                np.newaxis, ...]) * receiver_weights[..., np.newaxis])
    scattering_factor = 1 - scattering_flattened[np.newaxis, ...]
    brdf[:, :, :] += (
        scattering_flattened) / np.pi
    i_sources = np.arange(source_directions.csize)
    brdf[i_sources, i_receiver, :] += scattering_factor / cos_factor[
        i_sources, i_receiver, np.newaxis]

    brdf *= (1 - absorption_coefficient.freq.flatten())
    if file_path is not None:
        sofa = _create_sofa(
            pf.FrequencyData(brdf, scattering_coefficient.frequencies),
            source_directions,
            receiver_directions,
            history='constructed brdf based on scattering coefficients',
        )

        sf.write_sofa(file_path, sofa)
    return pf.FrequencyData(brdf, scattering_coefficient.frequencies)


def create_from_directional_scattering(
        source_directions,
        receiver_directions,
        directional_scattering,
        absorption_coefficient=None,
        file_path=None,
        ):
    r"""Create the BRDF from the directional scattering and write to SOFA file.

    The directional scattering coefficient is assumed to be anisotropic.
    The sum of the directional scattering coefficient has be equal to 1.
    Therefore the BRDF is calculated as follows:

    .. math::
        \rho(\mathbf{\Omega_i}, \mathbf{\Omega_o}) = \frac{(1-\alpha)}{
        (\mathbf{\Omega_o} \cdot \mathbf{n}) \cdot w_o} s_{d}(
        \mathbf{\Omega_i}, \mathbf{\Omega_o})

    where:
        - :math:`\mathbf{\Omega_i}` and :math:`\mathbf{\Omega_o}` are the
          incident and outgoing directions, respectively.
        - :math:`s_{d}` is the directional scattering coefficient [#]_.
        - :math:`\alpha` is the absorption coefficient.
        - :math:`\mathbf{n}` is the normal vector of the surface.
        - :math:`w_o` is weighting factor of the outgoing angular sector
          (unit sphere).

    Note that the weights doesn't need to be normalized,
    they get scaled as required.

    Parameters
    ----------
    source_directions : :py:class:`~pyfar.classes.coordinates.Coordinates`
        source directions for the BRDF, should contain weights.
        cshape of data should be (n_sources)
    receiver_directions : :py:class:`~pyfar.classes.coordinates.Coordinates`
        receiver directions for the BRDF, should contain weights.
        cshape of data should be (n_receivers)
    directional_scattering : :py:class:`~pyfar.classes.audio.FrequencyData`
        frequency dependent directional scattering coefficient data from [1]_.
        cshape of data should be (n_sources, n_receivers).
    absorption_coefficient : :py:class:`~pyfar.classes.audio.FrequencyData`
        frequency dependent absorption coefficient data, by default
        no absorption.
        cshape of data should be (1, ).
    file_path : string, path, optional
        path where sofa file should be saved, by default no file is saved.

    References
    ----------
    .. [#] A. Heimes and M. Vorländer, “Bidirectional surface scattering
           coefficients,” Acta Acust., vol. 9, p. 41, 2025,
           doi: 10.1051/aacus/2025026.



    """
    if not isinstance(source_directions, pf.Coordinates):
        raise TypeError(
            'source_directions must be a pf.Coordinates object')
    if not isinstance(receiver_directions, pf.Coordinates):
        raise TypeError(
            'receiver_directions must be a pf.Coordinates object')
    if (
            not isinstance(directional_scattering, pf.FrequencyData) or
            not directional_scattering.cshape == (
                source_directions.csize, receiver_directions.csize)):
        raise TypeError(
            'directional_scattering must be a pf.FrequencyData object with'
            f' cshape ({source_directions.csize, receiver_directions.csize})')
    if absorption_coefficient is None:
        absorption_coefficient = pf.FrequencyData(
            np.zeros_like(directional_scattering.frequencies),
            directional_scattering.frequencies)
    cos_receiver = np.cos(receiver_directions.colatitude)[
        np.newaxis, :, np.newaxis]
    receiver_weights = receiver_directions.weights
    receiver_weights *= 2 * np.pi / np.sum(receiver_weights)
    receiver_factor = receiver_weights[..., np.newaxis]
    brdf = directional_scattering.freq / receiver_factor / cos_receiver

    brdf *= (1 - absorption_coefficient.freq.flatten())
    if file_path is not None:
        sofa = _create_sofa(
            pf.FrequencyData(brdf, directional_scattering.frequencies),
            source_directions,
            receiver_directions,
            history='constructed brdf based on directional scattering',
        )

        sf.write_sofa(file_path, sofa)
    return pf.FrequencyData(brdf, absorption_coefficient.frequencies)


def _create_sofa(
    data,
    sources,
    receivers,
    history,
):
    """Write complex pressure to a SOFA object.

    Note that it will also write down the weights for the sources and
    the receivers.

    Parameters
    ----------
    data : numpy array
        The data as an array of shape (MRE)
    sources : pf.Coordinates
        source positions containing weights.
    receivers : pf.Coordinates
        receiver positions containing weights.
    history : string
        GLOBAL_History tag in the SOFA file.


    Returns
    -------
    sofa : sofar.Sofa object
        SOFA object with the data written to it

    """
    # create empty SOFA object
    convention = (
        'GeneralTF' if type(data) is pf.FrequencyData else 'GeneralFIR'
    )

    sofa = sf.Sofa(convention)

    # write meta data
    sofa.GLOBAL_ApplicationName = 'sparrowpy'
    sofa.GLOBAL_ApplicationVersion = sparrowpy.__version__
    sofa.GLOBAL_History = history

    # Source and receiver data
    sofa.EmitterPosition = sources.cartesian
    sofa.EmitterPosition_Units = 'meter'
    sofa.EmitterPosition_Type = 'cartesian'

    sources_sph = sources.spherical_elevation
    sources_sph = pf.rad2deg(sources_sph)
    sofa.SourcePosition = sources_sph
    sofa.SourcePosition_Units = 'degree, degree, metre'
    sofa.SourcePosition_Type = 'spherical'

    sofa.ReceiverPosition = receivers.cartesian
    sofa.ReceiverPosition_Units = 'meter'
    sofa.ReceiverPosition_Type = 'cartesian'

    if type(data) is pf.FrequencyData:
        sofa.N = data.frequencies

        # HRTF/HRIR data
        if data.cshape[0] != sources.csize:
            data.freq = np.swapaxes(data.freq, 0, 1)
        sofa.Data_Real = np.real(data.freq)
        sofa.Data_Imag = np.imag(data.freq)
    else:
        sofa.Data_IR = data.time
        sofa.Data_SamplingRate = data.sampling_rate
        sofa.Data_Delay = np.zeros((1, receivers.csize))

    sofa.add_variable('ReceiverWeights', receivers.weights, 'double', 'R')
    sofa.add_variable('SourceWeights', sources.weights, 'double', 'E')

    return sofa
