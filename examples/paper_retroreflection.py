# %%
import sparrowpy as sp
import numpy as np
import os
import pyfar as pf
import sofar as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyrato

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


plot_path = os.path.join(root_dir, 'jasa_paper', 'figures')
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# %%
height = 9
width = 19
# %%
source = pf.Coordinates(10+width/2, 10, 1.5)
receiver = pf.Coordinates(11+width/2, 11, 1.5)
patch_size = .25

sampling_rate = 1000
speed_of_sound = 343.2
etc_duration = .5  # seconds
etc_time = etc_duration


brdf_new, directions_bsc, directions_bsc = pf.io.read_sofa(
    os.path.join(root_dir, 'resources', 'brdf_new.sofa'))

brdf_rand, directions_bsc, directions_bsc = pf.io.read_sofa(
    os.path.join(root_dir, 'resources', 'brdf_rand.sofa'))

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

raven = np.loadtxt(os.path.join(root_dir, 'out', 'raven_facade.csv'), delimiter=",")

print(raven[0, 3:-2])
raven = pf.TimeData(
    raven[1:, 3:-2].T,
    raven[1:, 0],
)
frequencies_nom, frequencies_out = pf.dsp.filter.fractional_octave_frequencies(
    1, (np.min(brdf.frequencies)/1.1, np.max(brdf.frequencies)*1.1),
)


# %%
for is_decay in [True, False]:
    for i_band in range(brdf.n_bins):

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
            label='BSC-based BRDF, vertical profile',
            color='C1',
        )
        ax = pf.plot.time(
            edcs[1],
            dB=True, log_prefix=10, unit='ms', log_reference=1,
            label='BSC-based BRDF, horizontal profile',
            color='C2',
            linestyle='--',
        )
        ax = pf.plot.time(
            edcs[2],
            dB=True, log_prefix=10, unit='ms', log_reference=1,
            label='RISC-based BRDF',
            color='C0',
            linestyle=':',
        )
        ax = pf.plot.time(
            edcs[3],
            dB=True, log_prefix=10, unit='ms', log_reference=1,
            label='RAVEN reference (RISC)',
            color='C3',
            linestyle='--',
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
        plt.legend(fontsize=8,loc="upper right",bbox_to_anchor=(1.1, 1.2),shadow=True)
        fig.savefig(

            os.path.join(plot_path, f'facade_{str_fig}_{frequency_str}.pdf'),
            bbox_inches='tight',
        )
# %%

