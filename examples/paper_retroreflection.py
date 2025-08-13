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

def create_fig():
    figure,ax = plt.subplots(figsize=(3,2))
    plt.grid()
    return figure, ax

def create_fig2():
    figure,ax = plt.subplots(figsize=(5,3))
    plt.grid()
    return figure, ax

def export_fig(fig, filename,out_dir=root_dir, fformat=".pdf"):
    fig.savefig(os.path.join(out_dir,filename+fformat), bbox_inches='tight')

tlabel="$$rt \\quad [\\mathrm{s}]$$"
mlabel="peak memory [MiB]"
mlegend=["baking", "propagation","collection"]
tlegend=["baking", "propagation","collection","total"]

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
    spec_rand = random(
        bsc[np.arange(bsc_sources.csize), idx_spec],
        bsc_sources)

    retro_rand = random(
        bsc[np.arange(bsc_sources.csize), idx_retro],
        bsc_sources)

    fig = plt.figure()
    ax = pf.plot.freq(
        bsc[np.arange(bsc_sources.csize)[mask], idx_spec[mask]],
        dB=False, color='C2', linestyle=':',
        label='other incident directions')
    ax = pf.plot.freq(
        bsc[np.arange(bsc_sources.csize)[i_45], idx_spec[i_45]], 
        dB=False, color='C1', label='45° incidence')
    ax = pf.plot.freq(spec_rand, dB=False, color='C0', label='random incidence')

    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[-3:], labels[-3:],
    )
    ax.set_ylim((-0.05, 1))
    ax.set_ylabel('energy ratio into specular direction')
    if name is not None:
        fig.savefig(
            os.path.join(plot_path, f'{name}_specular.pdf'),
            bbox_inches='tight'
        )   

    print(mask)
    fig = plt.figure()
    ax = pf.plot.freq(
        bsc[np.arange(bsc_sources.csize)[mask], idx_retro[mask]],
        dB=False, color='C2', linestyle=':',
        label='single incident directions')
    ax = pf.plot.freq(
        bsc[np.arange(bsc_sources.csize)[i_45], idx_retro[i_45]], 
        dB=False, color='C1', label='45° incidence')
    ax = pf.plot.freq(
        retro_rand, dB=False, color='C0',
        label='random incidence')
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[-3:], labels[-3:],
    )
    # ax.set_ylim((-0.05, .05))
    ax.set_ylim((-0.05, 1))
    ax.set_ylabel('energy ratio into retro-reflection direction')
    # ax.set_ylabel('energy ratio into retro-reflection direction')
    if name is not None:
        fig.savefig(
            os.path.join(plot_path, f'{name}_retroreflection.pdf'),
            bbox_inches='tight'
        )

calc_plot_srand(bsc_sources, bsc_receivers, bsc, 'bsc')

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

calc_plot_srand(directions_bsc, directions_bsc, bsc_octave, 'bsc_mirrored_reduces_octave')


# calc_plot_srand(directions_bsc, directions_bsc, bsc_octave)

bsc_spec = directions_bsc.copy()
idx_retro = directions_bsc.find_nearest(bsc_spec)[0]
bsc_spec.azimuth += np.pi
idx_spec = directions_bsc.find_nearest(bsc_spec)[0]

s_rand = random(
    1-bsc_octave[np.arange(directions_bsc.csize), idx_spec],
    directions_bsc)

retro_rand = random(
    bsc_octave[np.arange(directions_bsc.csize), idx_retro],
    directions_bsc)
bsc_octave.freq /= np.sum(bsc_octave.freq, axis=1, keepdims=True)

# %%
height = 9
width = 19
# hight = 19
# # width = 9
# hight = 10
# width = 10

# length = 2
# width = 2


# %%

# %%

# %%
source = pf.Coordinates(10+width/2, 10, 1.5)
receiver = pf.Coordinates(11+width/2, 11, 1.5)
patch_size = .25
# source = pf.Coordinates(10, 10, 0)
# receiver = pf.Coordinates(11, 11, 0)
# source = pf.Coordinates(0, 10, 10)
# receiver = pf.Coordinates(0, 11, 11)

sampling_rate = 1000
speed_of_sound = 343.2
etc_duration = .5  # seconds
etc_time = etc_duration

# simulation parameters
brdf_rand = sp.brdf.create_from_scattering(
    directions_bsc,
    directions_bsc,
    s_rand,
    pf.FrequencyData(np.zeros_like(bsc_octave.frequencies), bsc_octave.frequencies),
)

brdf_new = sp.brdf.create_from_directional_scattering(
    directions_bsc,
    directions_bsc,
    bsc_octave,
    pf.FrequencyData(np.zeros_like(bsc_octave.frequencies), bsc_octave.frequencies),
)

edcs = []
brdfs = [brdf_new, brdf_new, brdf_rand]
for i_brdf in range(len(brdfs)):
    brdf = brdfs[i_brdf]
    if i_brdf == 0:
        up_vector_wall = [1, 0, 0]
    else:
        up_vector_wall = [0, 0, 1]

    plane = sp.geometry.Polygon(
            [[0, 0, 0,],
            [width, 0, 0],
            [width, 0, height],
            [0, 0, height]],
            up_vector_wall, [0, 1, 0])

    #simulation parameters
    radi = sp.DirectionalRadiosityFast.from_polygon(
        [plane], patch_size)


    radi.set_wall_brdf(
        np.arange(1), brdf, directions_bsc, directions_bsc)

    # set air absorption
    radi.set_air_attenuation(
        pf.FrequencyData(
            np.zeros_like(brdf.frequencies),
            brdf.frequencies))

    # initialize source energy at each patch
    radi.init_source_energy(source)

    # # gather energy at receiver
    radi.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        etc_time_resolution=1/sampling_rate,
        etc_duration=etc_duration,
        max_reflection_order=0)
    
    direct_sound, _ = radi.calculate_direct_sound(receiver)

    edc = radi.collect_energy_receiver_mono(receiver, True)
    edcs.append(edc)

edc_new_vertical = edcs[0]
edc_new_horizontal = edcs[1]
edc_rand = edcs[2]

# %%
plane_dim = np.array(
    [
        [0, 0, 0],
        [width, 0, 0],
        [width, 0, height],
        [0, 0, height],
     ])
X, Z = np.meshgrid(
        np.linspace(0, width, 2),
        np.linspace(0, height, 2),
    )
Y = np.zeros_like(X)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.quiver(0,0,0, 24,0,0, color='k', arrow_length_ratio=0.1/2,)
ax.quiver(0,0,0, 0,12,0, color='k', arrow_length_ratio=0.1,)
ax.quiver(0,0,0, 0,0,12, color='k', arrow_length_ratio=0.1,)
ax.text(25, 0, 0, 'x', color='k', horizontalalignment='center', verticalalignment='center')
ax.text(0, 13, 0, 'y', color='k', horizontalalignment='center', verticalalignment='center')
ax.text(0, 0, 13, 'z', color='k', horizontalalignment='center', verticalalignment='center')

Lambda = 0.08*8
for i in range(29):
    ax.plot(np.zeros(2)+(i+1)*Lambda, 0, [0, height], color='C1', linestyle='-', linewidth=0.3)

# for i in range(13):
#     ax.plot([0, width], 0, np.zeros(2)+(i+1)*Lambda, color='C2', linestyle='-', linewidth=0.3)

ax.plot(plane_dim[:, 0], plane_dim[:, 1], plane_dim[:, 2], color='C0', label='facade')
ax.scatter(width, 0, height, color='C0')
ax.text(width, 0, height+1, f'({width:.0f}, {0:.0f}, {height:.0f})', color='C0', horizontalalignment='center', verticalalignment='center')

# ax.plot_surface(X, Y, Z, color='C0', label='facade')

sx = source.x[0]
sy = source.y[0]
sz = source.z[0]
rx = receiver.x[0]
ry = receiver.y[0]
rz = receiver.z[0]
ax.scatter(sx, sy, sz, color='C3', label='source')
ax.plot([0, sx], [sy, sy], [0, 0],color='C3', linestyle='--')
ax.plot([sx, sx], [0, sy], [0, 0],color='C3', linestyle='--')
ax.plot([sx, sx], [sy, sy], [0, sz], color='C3', linestyle='--')
ax.text(sx-.6, sy, sz, f'({sx:.1f}, {sy:.1f}, {sz:.1f})', color='C3', horizontalalignment='left', verticalalignment='center')

ax.scatter(rx, ry, rz, color='C4', label='receiver')
ax.plot([0, rx], [ry, ry], [0, 0], color='C4', linestyle='--')
ax.plot([rx, rx], [ry, ry], [0, rz], color='C4', linestyle='--')
ax.plot([rx, rx], [0, ry], [0, 0],color='C4', linestyle='--')
ax.text(rx, ry+3.5, rz, f'({rx:.1f}, {ry:.1f}, {rz:.1f})', color='C4', horizontalalignment='center', verticalalignment='center')

ax.set_aspect('equal')
ax.set_axis_off()
ax.view_init(elev=30, azim=90, roll=0)
fig.savefig(
    os.path.join(plot_path, 'facade_3d.pdf'),
    bbox_inches='tight',
    transparent=True,
)
# %%

# %%

# %%
i_band=-1
i_receiver = 0
energy_direct_db = 10*np.log10(direct_sound/1e-12)

plt.figure()

all_etcs =  pf.utils.concatenate_channels([
        edc_new_vertical[i_receiver, i_band],
        edc_rand[i_receiver, i_band],
        edc_new_horizontal[i_receiver, i_band],
    ])
energy = np.sum(all_etcs.time, axis=-1)
energy_diff_dB = 10*np.log10(energy/direct_sound[0, i_band])
energy_dB = 10*np.log10(energy/1e-12)
ax = pf.plot.time(
    all_etcs, dB=True, log_prefix=10, unit='s', log_reference=1e-12,)
ax.legend([
    f'BSC vertical dL={energy_dB[0]:.1f} dB',
    f'Random scattering dL={energy_dB[1]:.1f} dB',
    f'BSC horizontal dL={energy_dB[2]:.1f} dB',
])
ax.set_xlim((0, 0.15))
ax.set_title(f'Frequency: {bsc_octave.frequencies[i_band]} Hz')
# %%

raven = np.loadtxt(os.path.join(root_dir, 'out', 'raven_facade.csv'), delimiter=",")

print(raven[0, 3:-2])
raven = pf.TimeData(
    raven[1:, 3:-2].T,
    raven[1:, 0],
)


# %%
for is_decay in [True, False]:
    for i_band in range(bsc_octave.n_bins):

        i_receiver = 0
        energy_direct_db = 10*np.log10(direct_sound/1e-12)

        all_etcs =  pf.utils.concatenate_channels([
                edc_new_vertical[i_receiver, i_band],
                edc_new_horizontal[i_receiver, i_band],
                edc_rand[i_receiver, i_band],
                raven[i_band]/(4*np.pi),
            ])
        if is_decay:
            edcs = pyrato.edc.schroeder_integration(all_etcs, True)
        else:
            edcs = all_etcs
        fig, ax = create_fig2()
        ax = pf.plot.time(
            edcs[0],
            dB=True, log_prefix=10, unit='ms', log_reference=1,
            label='BSC vertical',
            color='C1',
        )
        ax = pf.plot.time(
            edcs[2],
            dB=True, log_prefix=10, unit='ms', log_reference=1,
            label='BSC horizontal',
            color='C2',
        )
        ax = pf.plot.time(
            edcs[1],
            dB=True, log_prefix=10, unit='ms', log_reference=1,
            label='Random scattering',
            color='C0',
        )
        ax = pf.plot.time(
            edcs[3],
            dB=True, log_prefix=10, unit='ms', log_reference=1,
            label='Raven',
            color='C3',
            linestyle='-',
        )

        ax.set_xlim((0, 150))

        ax.set_xlabel("Time [ms]")
        if is_decay:
            ax.set_ylabel("Energy decay curve [dB]")
        else:
            ax.set_ylabel("Energy time curve [dB]")
        ax.set_ylim([-150+40,-40+40])
        ff = frequencies_nom[i_band]
        frequency_str = f'{ff/1000:.0f}kHz' if ff >=1e3 else f'{ff:.0f}Hz'
        str_fig = 'edc' if is_decay else 'etc'
        fig.savefig(

            os.path.join(plot_path, f'facade_{str_fig}_{frequency_str}.pdf'),
            bbox_inches='tight',
        )
        plt.legend(fontsize=8)

        export_fig(fig,filename="etcs_500")
# %%
