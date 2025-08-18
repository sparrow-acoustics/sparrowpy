# %%
import sparrowpy as sp
import numpy as np
import os
import pyfar as pf
import sofar as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyrato
# %matplotlib ipympl

# %%

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

# %%
# read simulation results
root_dir=os.path.join(os.getcwd())


font={
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Helvetica",
    "font.size": 12,
}



plt.rcParams.update(font)


def create_fig2():
    figure,ax = plt.subplots(figsize=(3,2))
    plt.grid()
    return figure, ax

# %%
# set up the simulation parameters
sofa_brdf = sf.read_sofa(
    os.path.join(root_dir, 'resources', 'triangle_sim_optimal.s_d.sofa'))
bsc, bsc_sources, bsc_receivers = pf.io.convert_sofa(sofa_brdf)
bsc_sources.weights = sofa_brdf.SourceWeights
bsc_receivers.weights = sofa_brdf.ReceiverWeights

plot_path = os.path.join(root_dir, 'jasa_paper', 'figures')
if not os.path.exists(plot_path):
    os.makedirs(plot_path)


def calc_plot_srand(bsc_sources, bsc_receivers, bsc, name=None):
    bsc_spec = bsc_sources.copy()
    i_45 = bsc_spec.find_nearest(
        pf.Coordinates.from_spherical_elevation(
            0, np.pi/4, bsc_spec.radius[0]),
        distance_measure='spherical_radians',
        radius_tol=1e-13)[0][0]
    idx_retro = bsc_receivers.find_nearest(bsc_spec)[0][0]
    bsc_spec.azimuth += np.pi
    idx_spec = bsc_receivers.find_nearest(bsc_spec)[0][0]
    mask = (bsc_sources.azimuth < 1*np.pi/180) | (bsc_sources.azimuth > 359*np.pi/180)
    s_rand = random(
        1-bsc[np.arange(bsc_sources.csize), idx_spec],
        bsc_sources)

    retro_rand = random(
        bsc[np.arange(bsc_sources.csize), idx_retro],
        bsc_sources)

    fig, ax = create_fig2()
    ax = pf.plot.freq(
        1-bsc[np.arange(bsc_sources.csize)[mask], idx_spec[mask]],
        dB=False, color='C0', linestyle=':',
        label='single incidence')
    # ax = pf.plot.freq(
    #     1-bsc[np.arange(bsc_sources.csize)[i_45], idx_spec[i_45]], 
    #     dB=False, color='C1', label='45° incidence')
    ax = pf.plot.freq(s_rand, dB=False, color='C0', label='random incidence')

    ax.legend(fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[-2:], labels[-2:],
    )
    ax.set_ylim((-0.05, 1))
    ax.set_ylabel('scattering coefficient')
    if name is not None:
        fig.savefig(
            os.path.join(plot_path, f'{name}_specular.pdf'),
            bbox_inches='tight',
        )
    return s_rand
    # print(mask)
    # fig, ax = create_fig2()
    # ax = pf.plot.freq(
    #     bsc[np.arange(bsc_sources.csize)[mask], idx_retro[mask]],
    #     dB=False, color='C2', linestyle=':',
    #     label='single incident directions')
    # ax = pf.plot.freq(
    #     bsc[np.arange(bsc_sources.csize)[i_45], idx_retro[i_45]], 
    #     dB=False, color='C1', label='45° incidence')
    # ax = pf.plot.freq(
    #     retro_rand, dB=False, color='C0',
    #     label='random incidence')
    # ax.legend()
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(
    #     handles[-3:], labels[-3:],
    # )
    # # ax.set_ylim((-0.05, .05))
    # ax.set_ylim((-0.05, 1))
    # ax.set_ylabel('energy ratio into retro-reflection direction')
    # # ax.set_ylabel('energy ratio into retro-reflection direction')
    # if name is not None:
    #     fig.savefig(
    #         os.path.join(plot_path, f'{name}_retroreflection.pdf'),
    #         bbox_inches='tight'
        # )

s_rand_orig = calc_plot_srand(bsc_sources, bsc_receivers, bsc, 'bsc')
s_rand_orig._frequencies /= 8

# # %%
# plt.figure()
# ax = pf.plot.freq(bsc[np.arange(11), idx_spec[:11]], dB=False, color='C0', linestyle='--')
# ax = pf.plot.freq(s_rand, dB=False, color='C1')
# ax
# ax.set_ylim((0, 1))

# plt.figure()
# ax = pf.plot.freq(bsc[np.arange(11), idx_retro[:11]], dB=False, color='C0', linestyle='--')
# ax = pf.plot.freq(retro_rand, dB=False, color='C1')
# ax
# ax.set_ylim((0, 1))
#%%

bsc_incident_directions = bsc_receivers.copy()
bsc_scattering_directions = bsc_receivers.copy()
bsc_mirrored = pf.FrequencyData(
    np.zeros((bsc_incident_directions.csize, bsc_scattering_directions.csize, bsc.n_bins)),
    bsc.frequencies)
for i_source in tqdm(range(bsc_receivers.csize)):
    # mirror results from previous directions for Azimuth > 90°
    if bsc_incident_directions[i_source].azimuth > np.pi/2:
        az_is = bsc_incident_directions[i_source].azimuth[0]
        if az_is < np.pi:  # 180:
            az_mirror = np.pi - az_is
        elif az_is < 3/2*np.pi:  # 270°
            az_mirror = np.abs(np.pi - az_is)
        else:
            az_mirror = 2*np.pi - az_is
        # print(f'Azimuth {az_is*180/np.pi:.1f} mirrored to {az_mirror*180/np.pi:.1f}')
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
        bsc_mirrored.freq[i_source, :, :] = bsc.freq[i_source_mirror, idx_scattering, :]
    else:       
        bsc_mirrored.freq[i_source, :, :] = bsc.freq[i_source, :, :]

bsc_sources = bsc_receivers.copy()
bsc_mirrored._frequencies /= 8

calc_plot_srand(bsc_sources, bsc_receivers, bsc_mirrored)


# %%
# reduction

directions_bsc = pf.samplings.sph_gaussian(sh_order=11)
directions_bsc = directions_bsc[directions_bsc.z>0]

bsc_mirrored_reduced = pf.FrequencyData(
    np.zeros((directions_bsc.csize, directions_bsc.csize, bsc_mirrored.n_bins)),
    bsc_mirrored.frequencies)
n_directions = np.zeros((directions_bsc.csize, directions_bsc.csize, 1))

for i_source in tqdm(range(bsc_incident_directions.csize)):
    i_source_nearest = directions_bsc.find_nearest(
        bsc_incident_directions[i_source], radius_tol=1e-13)[0][0]
    for i_receiver in range(bsc_incident_directions.csize):
        i_receiver_nearest = directions_bsc.find_nearest(
            bsc_incident_directions[i_receiver], radius_tol=1e-13)[0][0]
    
        bsc_mirrored_reduced.freq[
            i_source_nearest, i_receiver_nearest, :] += bsc_mirrored.freq[
                i_source, i_receiver, :]
        n_directions[i_source_nearest, i_receiver_nearest] += 1

bsc_mirrored_reduced.freq /= n_directions
assert not np.any(n_directions==0)

calc_plot_srand(directions_bsc, directions_bsc, bsc_mirrored_reduced)
# %%
frequencies_nom, frequencies_out = pf.dsp.filter.fractional_octave_frequencies(
    1, (np.min(bsc_mirrored.frequencies), np.max(bsc_mirrored.frequencies)),
)

bsc_octave = average_frequencies(
    bsc_mirrored_reduced, frequencies_out, domain='energy')

s_rand_orig_oct = average_frequencies(
    s_rand_orig, frequencies_out, domain='energy')

s_rand_oct = calc_plot_srand(directions_bsc, directions_bsc, bsc_octave, 'bsc_mirrored_reduces_octave')


# calc_plot_srand(directions_bsc, directions_bsc, bsc_octave)

bsc_spec = directions_bsc.copy()
idx_retro = directions_bsc.find_nearest(bsc_spec)[0]
bsc_spec.azimuth += np.pi
idx_spec = directions_bsc.find_nearest(bsc_spec)[0]

bsc_octave.freq /= np.sum(bsc_octave.freq, axis=1, keepdims=True)


# %%

abs_wall = pf.FrequencyData(
    .07*np.ones_like(bsc_octave.frequencies), bsc_octave.frequencies)
brdf_rand = sp.brdf.create_from_scattering(
    directions_bsc,
    directions_bsc,
    s_rand_orig_oct,
    abs_wall,
)

brdf_new = sp.brdf.create_from_directional_scattering(
    directions_bsc,
    directions_bsc,
    bsc_octave,
    abs_wall,
)


brdf_ground = sp.brdf.create_from_scattering(
    directions_bsc,
    directions_bsc,
    pf.FrequencyData(np.ones_like(bsc_octave.frequencies), bsc_octave.frequencies),
    pf.FrequencyData(.01*np.ones_like(bsc_octave.frequencies), bsc_octave.frequencies))
# %%

sofa_brdf_rand = sf.Sofa('GeneralTF')

sofa_brdf_rand = add_source_receiver_data(
    sofa_brdf_rand, directions_bsc, directions_bsc, brdf_rand)
sf.write_sofa(os.path.join(root_dir, 'resources', 'brdf_rand.sofa'), sofa_brdf_rand)


sofa_brdf_new = sf.Sofa('GeneralTF')
sofa_brdf_new = add_source_receiver_data(
    sofa_brdf_new, directions_bsc, directions_bsc, brdf_new)
sf.write_sofa(os.path.join(root_dir, 'resources', 'brdf_new.sofa'), sofa_brdf_new)


sofa_brdf_ground = sf.Sofa('GeneralTF')
sofa_brdf_ground = add_source_receiver_data(
    sofa_brdf_ground, directions_bsc, directions_bsc, brdf_ground)
sf.write_sofa(os.path.join(root_dir, 'resources', 'brdf_ground.sofa'), sofa_brdf_ground)




# %%
fig, ax = create_fig2()
# ax = pf.plot.freq(
#     s_rand_orig,
#     dB=False, color='C0', linestyle=':',
#     label='single incidence')
ax = pf.plot.freq(
    s_rand_orig_oct,
    dB=False, color='C0', linestyle='-',
    label='RISC')
ax.legend(fontsize=8)
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[-2:], labels[-2:],
)
ax.set_ylim((-0.05, 1))
ax.set_ylabel('$s_\mathrm{rand}$')
fig.savefig(

    os.path.join(plot_path, 's_rand.pdf'),
    bbox_inches='tight',
)








# %%
# plot brdf polar

incident_direction = pf.Coordinates.from_spherical_colatitude(
    0, np.pi/4, 1)

i_in = directions_bsc.find_nearest(incident_direction)[0]

i_out = np.where(directions_bsc.y==0)[0]
i_out = i_out[np.argsort(directions_bsc[i_out].upper)]

for iband in [-1]:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    delta_angle = 0.25094759
    ax.plot(
        [3*np.pi/4, 3*np.pi/4], [19, 16],
        color='C3', alpha=1,
        label='Incident direction',
    )
    for i_an in range(len(i_out)):
        center = np.pi-directions_bsc[i_out[i_an]].upper[0]
        lower = center - delta_angle/2
        upper = center + delta_angle/2

        brdf = brdf_rand[i_in, i_out]
        norm = np.max(brdf.freq[:, iband])
        norm = .2
        old = brdf.freq[i_an, iband] / norm
        ax.plot(
            [lower, upper], [old, old],
            label=r'5$\times$RISC-based BRDF', c='C1')
        ax.plot(
            [lower, lower], [0, old],
            label='RISC-based BRDF', c='C1')
        ax.plot(
            [upper, upper], [0, old],
            label='RISC-based BRDF', c='C1')

    for i_an in range(len(i_out)):
        center = np.pi-directions_bsc[i_out[i_an]].upper[0]
        lower = center - delta_angle/2
        upper = center + delta_angle/2
        brdf = brdf_new[i_in, i_out]
        norm = np.max(brdf.freq[:, iband])
        norm = 1
        new = brdf.freq[i_an, iband] / norm
        ax.plot(
            [lower, upper], [new, new],
            label='BSC-based BRDF', c='C0', linestyle='-')
        ax.plot(
            [lower, lower], [0, new],
            label='BSC-based BRDF', c='C0', linestyle='-')
        ax.plot(
            [upper, upper], [0, new],
            label='BSC-based BRDF', c='C0', linestyle='-')

    ax.plot(
        [0, np.pi], [10, 10],
        color='k', alpha=1,
        label='Incident direction',
    )
    ax.arrow(
        3*np.pi/4, 19, 0, -3,
        color='C3', alpha=1,
        head_width=.07,
        head_length=1.2,
        linewidth=1,
        label='Incident direction',
    )
    # ax.arrow(
    #     3*np.pi/4, 19, 0, 0,
    #     color='C3', alpha=1,
    #     head_width=.1,
    #     head_length=1,
    #     linewidth=1,
    #     label='Incident direction',
    # )
    ax.legend(fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    handles = np.array(handles)
    labels = np.array(labels)
    mask = np.zeros(len(handles), dtype=bool)
    mask[1] = True
    mask[-5] = True
    mask[0] = True
    ax.legend(
        handles[mask], labels[mask], loc='right', bbox_to_anchor=(1.6, 0.6),
    )
    ff = frequencies_nom[iband]
    frequency_str = f'{ff/1000:.0f}kHz' if ff >=1e3 else f'{ff:.0f}Hz'
    ax.set_axis_off()
    fig.savefig(
        os.path.join(plot_path, f'BRDF_{frequency_str}.pdf'),
        bbox_inches='tight', pad_inches=-0.1,
    )
# %%

