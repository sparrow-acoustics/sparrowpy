import pyfar as pf
import numpy as np
import os
import sofar as sf



def create_from_scattering(
        file_path, scattering_coefficient=1, sh_order=1, frequencies=[1000]):
    """Create a SOFA file from a scattering coefficient.

    Parameters
    ----------
    file_path : string, path
        path where sofa file should be saved
    scattering_coefficient : int, optional
        broadcastable to , by default 1
    sh_order : int, optional
        sh order of the gaussian sampling of the sources and receivers
        coordinates, by default 1
    frequencies : list, optional
        frequency bins for the directivity, by default [1000]

    """
    shape = np.broadcast_shapes(
        np.asarray(scattering_coefficient).shape,
        np.asarray(frequencies).shape,
        (1,))
    if len(shape) == 1:
        raise TypeError('scattering_coefficient and frequencies must '\
        'be broadcastable to a 1D array')
    scattering_coefficient = np.broadcast_to(scattering_coefficient, shape)
    frequencies = np.broadcast_to(frequencies, shape)


    sources = pf.samplings.sph_gaussian(sh_order=sh_order)
    sources = sources[sources.z > 0]
    receivers = pf.samplings.sph_gaussian(sh_order=sh_order)
    receivers = receivers[receivers.z > 0]
    data_out = np.zeros((sources.csize, receivers.csize, frequencies.size))

    speed_of_sound = 343
    for i_source in range(sources.csize):
        source = sources[i_source]
        image_source = source.copy()
        image_source.azimuth += np.pi
        i_receiver = receivers.find_nearest(image_source)[0][0]
        data_out[i_source, :, :] = (
            scattering_coefficient) / np.pi
        data_out[i_source, i_receiver, :] += (
            1 - scattering_coefficient) / np.cos(source.elevation)


    sofa = _create_sofa(
        pf.FrequencyData(data_out, frequencies),
        sources,
        receivers,
        speed_of_sound=speed_of_sound,
        density_of_medium=1.2,
        Mesh2HRTF_version='0.1',
    )

    sf.write_sofa(file_path, sofa)



def _create_sofa(
    data,
    sources,
    receivers,
    speed_of_sound,
    density_of_medium,
    Mesh2HRTF_version,
):
    """Write complex pressure to a SOFA object.

    Parameters
    ----------
    data : numpy array
        The data as an array of shape (MRE)
    sources : _type_
        _description_
    receivers : _type_
        _description_
    speed_of_sound : _type_
        _description_
    density_of_medium : _type_
        _description_
    Mesh2HRTF_version : _type_
        _description_

    Returns
    -------
    sofa : sofar.Sofa object
        SOFA object with the data written to it

    """
    # create empty SOFA object
    convention = (
        'GeneralTF' if type(data) == pf.FrequencyData else 'GeneralFIR'
    )

    sofa = sf.Sofa(convention)

    # write meta data
    sofa.GLOBAL_ApplicationName = 'Mesh2scattering'
    sofa.GLOBAL_ApplicationVersion = Mesh2HRTF_version
    sofa.GLOBAL_History = 'numerically simulated data'

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

    if type(data) == pf.FrequencyData:
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

    sofa.add_variable('SpeedOfSound', speed_of_sound, 'double', 'I')
    sofa.add_variable('DensityOfMedium', density_of_medium, 'double', 'I')
    sofa.add_variable('ReceiverWeights', receivers.weights, 'double', 'R')
    sofa.add_variable('SourceWeights', sources.weights, 'double', 'E')

    return sofa
