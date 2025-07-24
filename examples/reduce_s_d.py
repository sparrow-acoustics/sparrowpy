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

def get_bsc(file_in = 'examples/resources/triangle_sim_optimal.s_d.sofa'):
    sofa = sf.read_sofa(file_in)
    bsc, sources, receivers = pf.io.convert_sofa(sofa)
    sources.weights = sofa.SourceWeights
    receivers.weights = sofa.ReceiverWeights

    print(bsc)

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
    bsc_mirrored.freq /= np.sum(bsc_mirrored.freq, axis=1)

    frequencies_out = pf.dsp.filter.fractional_octave_frequencies(
        1, (np.min(bsc.frequencies/8), np.max(bsc.frequencies/8)),
    )[1]

    bsc_octave = average_frequencies(
        bsc_mirrored, frequencies_out, domain='energy')

    return bsc_octave, sources, receivers


if __name__=="__main__":
    get_bsc()