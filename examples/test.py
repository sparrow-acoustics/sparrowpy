import sparapy as sp
import sparapy.geometry as geo
from sparapy.radiosity import DirectionalRadiosity
from sparapy.radiosity_fast import DRadiosityFast
import cProfile
import pyfar as pf
import sofar as sf
import os
import tqdm
from datetime import datetime
import numpy as np
import tracemalloc

sample_walls = sp.testing.shoebox_room_stub(1, 1, 1)

def init_energy(rad, source_pos):
    rad.source = geo.SoundSource(source_pos, [1, 0, 0], [0, 0, 1])
    E_matrix = []
    for patches in rad.patch_list:
        patches.init_energy_exchange(
            rad.max_order_k, rad.ir_length_s, rad.source,
            sampling_rate=rad.sampling_rate)
        E_matrix.append(patches.E_matrix)
    return E_matrix

def calc_form_factor(rad):
    if len(rad.patch_list) > 1:
        for patches in rad.patch_list:
            patches.calculate_form_factor(rad.patch_list)

def energy_exchange(rad):
    if len(rad.patch_list) > 1:
        for k in tqdm.tqdm(range(1, rad.max_order_k+1)):
            for patches in rad.patch_list:
                patches.calculate_energy_exchange(
                    rad.patch_list, k, speed_of_sound=rad.speed_of_sound,
                    E_sampling_rate=rad.sampling_rate)


gaussian = pf.samplings.sph_gaussian(sh_order=1)
gaussian = gaussian[gaussian.z>0]
sources = gaussian.copy()
receivers = gaussian.copy()
frequencies = pf.dsp.filter.fractional_octave_frequencies(
    1, (100, 1000))[0]
data = np.ones((sources.csize, receivers.csize, frequencies.size))
data = pf.FrequencyData(data, frequencies)
sofa = sf.Sofa('GeneralTF')
sofa.Data_Real = data.freq
sofa.Data_Imag = np.zeros_like(sofa.Data_Real)
sofa.N = data.frequencies
sofa.SourcePosition = sources.cartesian
sofa.ReceiverPosition = pf.rad2deg(receivers.spherical_elevation)
sofa_path = os.path.join(os.getcwd(), 'test.sofa')
sf.write_sofa(sofa_path, sofa)

n_max = 2
repeat = 1
steps_names = [
    'create patches', 'init energy', 'form factor',
    'energy exchange', 'collect energy',
    ]
energy_threshold = 1e-1

steps = len(steps_names)
fast_second = np.zeros((steps, n_max, repeat))
memory_fast = np.zeros((steps, n_max, repeat))
memory_slow = np.zeros((steps, n_max, repeat))
# memory_slow[:] = np.nan
slow = np.zeros((steps, n_max, repeat))
# slow[:] = np.nan

number_of_patches = np.zeros((n_max))

# # run one time, to get complied
# radios = DRadiosityFast.from_polygon(sample_walls,1.)

# radios.set_wall_scattering(
#     wall_indexes=np.arange(6), scattering=data, sources=sources, receivers=receivers)
# radios.set_air_attenuation(
#     pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
# radios.set_wall_absorption(
#     np.arange(6),
#     pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))

# radios.bake_geometry()

# radios.init_source_energy([0.5, 0.5, 0.5])
# histogram = radios.calculate_energy_exchange_receiver(
#     [0.5, 0.5, 0.5], 343, 1e-3, 1,
#     max_time=0.011)

# ACTUAL RUNS
max_order_k = 5
for method in ['queue']:
    for i in range(n_max):
        max_size = 1/(2**i)
        start_loop = datetime.now()
        print(f'{datetime.now()} run({i+1}/{n_max}): {max_size}')
        # run fast two times
        for j in [0]:
            # create patches and add material
            print("initializing radiosity....",end="")
            radiosity = DRadiosityFast.from_polygon(sample_walls, max_size)
            print("done!")
            
            print("setting up room characteristics....",end="")
            radiosity.set_wall_scattering(
                np.arange(6), data, sources, receivers)
            radiosity.set_air_attenuation(
                pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
            radiosity.set_wall_absorption(
                np.arange(6),
                pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
            print("done!")


            print("baking geometry (ff-tilde)......",end="")
            radiosity.bake_geometry()
            print("done!")

            print("initializing energy from source.....",end="")
            radiosity.init_source_energy([0.5, 0.5, 0.5], algorithm=method)
            print('done!')

            print("energy exchange........",end="")
            histogram = radiosity.calculate_energy_exchange_receiver(
                receiver_pos=np.array([0.5,0.5,0.5]), speed_of_sound=343.,
                histogram_time_resolution=1e-3, histogram_length=0.1, algorithm=method,
                threshold=energy_threshold)
            print("done!")

        number_of_patches[i] = radiosity.n_patches
        if i < 4:
            # Run old
            for j in range(repeat):
                # create patches
                tracemalloc.start()
                start = datetime.now()
                radiosity_old = DirectionalRadiosity(
                    sample_walls, max_size, max_order_k, 1, sofa_path)
                delta = (datetime.now() - start)
                slow[0, i, j] = (delta.seconds*1e6 + delta.microseconds)/repeat
                memory_slow[0, i, j] = tracemalloc.get_traced_memory()[1] # get peak memory
                tracemalloc.stop()

                # init_energy
                tracemalloc.start()
                start = datetime.now()
                E_matrix = init_energy(radiosity_old, [0.5, 0.5, 0.5])
                delta = (datetime.now() - start)
                slow[1, i, j] = (delta.seconds*1e6 + delta.microseconds)/repeat
                memory_slow[1, i, j] = tracemalloc.get_traced_memory()[1] # get peak memory
                tracemalloc.stop()

                # form factor
                tracemalloc.start()
                start = datetime.now()
                calc_form_factor(radiosity_old)
                delta = (datetime.now() - start)
                slow[2, i, j] = (delta.seconds*1e6 + delta.microseconds)/repeat
                memory_slow[2, i, j] = tracemalloc.get_traced_memory()[1] # get peak memory
                tracemalloc.stop()

                # energy exchange
                tracemalloc.start()
                start = datetime.now()
                energy_exchange(radiosity_old)
                delta = (datetime.now() - start)
                slow[3, i, j] = (delta.seconds*1e6 + delta.microseconds)/repeat
                memory_slow[3, i, j] = tracemalloc.get_traced_memory()[1] # get peak memory
                tracemalloc.stop()

                # collect energy
                tracemalloc.start()
                start = datetime.now()
                receiver = sp.geometry.Receiver([0.5, 0.5, 0.5], [1, 0, 0], [0, 0, 1])
                radiosity_old.energy_at_receiver(receiver)
                delta = (datetime.now() - start)
                slow[4, i, j] = (delta.seconds*1e6 + delta.microseconds)/repeat
                memory_slow[4, i, j] = tracemalloc.get_traced_memory()[1] # get peak memory
                tracemalloc.stop()
        delta = (datetime.now() - start_loop)
        delta_seconds = (delta.seconds*1e6 + delta.microseconds)*1e-6
        print(f'{datetime.now()}   took {delta_seconds} seconds')


# %%
import matplotlib.pyplot as plt
slow[slow == 0] = np.nan
# plot time to compute
plt.figure()
ax = plt.gca()
ax.semilogy(
    number_of_patches, np.mean(np.sum(fast_second, axis=0)*1e-6, axis=-1),
    label='fast implementation (second call)', marker='o')
ax.semilogy(
    number_of_patches, np.mean(np.sum(slow, axis=0)*1e-6, axis=-1),
    label='old implementation (classes)', marker='o')
ax.semilogy(
    number_of_patches, np.sum(fast_second, axis=0)*1e-6, color='C0', alpha=0.5)
ax.semilogy(
    number_of_patches, np.sum(slow, axis=0)*1e-6, color='C1', alpha=0.5)
ax.grid()
ax.set_xscale('log')
ax.set_xlabel('number of patches')
ax.set_ylabel('time [s]')
ax.set_title(f'overall (n={repeat})')
ax.set_ylim([1e-6, 1e2])
plt.legend()

for i in range(steps):
    plt.figure()
    ax = plt.gca()

    ax.semilogy(
        number_of_patches, np.mean(fast_second[i]*1e-6, axis=-1),
        label='fast implementation (second call)', marker='o')
    ax.semilogy(
        number_of_patches, np.mean(slow[i]*1e-6, axis=-1),
        label='old implementation (classes)', marker='o')
    ax.semilogy(
        number_of_patches, fast_second[i]*1e-6, color='C0', alpha=0.5)
    ax.semilogy(
        number_of_patches, slow[i]*1e-6, color='C1', alpha=0.5)
    ax.grid()
    ax.set_xscale('log')
    ax.set_xlabel('number of patches')
    ax.set_ylabel('time [s]')
    ax.set_title(f'{steps_names[i]} (n={repeat})')
    ax.set_ylim([1e-6, 1e2])
    plt.legend()

# %%
# plot memory consumption
memory_slow[memory_slow == 0] = np.nan

plt.figure()
ax = plt.gca()
ax.semilogy(
    number_of_patches, np.mean(np.sum(memory_fast, axis=0)*1e-6, axis=-1),
    label='fast implementation (second call)', marker='o')
ax.semilogy(
    number_of_patches, np.mean(np.sum(memory_slow, axis=0)*1e-6, axis=-1),
    label='old implementation (classes)', marker='o')
ax.semilogy(
    number_of_patches, np.sum(memory_fast, axis=0)*1e-6, color='C0', alpha=0.5)
ax.semilogy(
    number_of_patches, np.sum(memory_slow, axis=0)*1e-6, color='C1', alpha=0.5)
ax.grid()
ax.set_xscale('log')
ax.set_xlabel('number of patches')
ax.set_ylabel('memory [MB]')
ax.set_title(f'overall (n={repeat})')
ax.set_ylim([1e-6, 1e2])
plt.legend()

for i in range(steps):
    plt.figure()
    ax = plt.gca()

    ax.semilogy(
        number_of_patches, np.mean(memory_fast[i]*1e-6, axis=-1),
        label='fast implementation (second call)', marker='o')
    ax.semilogy(
        number_of_patches, np.mean(memory_slow[i]*1e-6, axis=-1),
        label='old implementation (classes)', marker='o')
    ax.semilogy(
        number_of_patches, memory_fast[i]*1e-6, color='C0', alpha=0.5)
    ax.semilogy(
        number_of_patches, memory_slow[i]*1e-6, color='C1', alpha=0.5)
    ax.grid()
    ax.set_xscale('log')
    ax.set_xlabel('number of patches')
    ax.set_ylabel('memory [MB]')
    ax.set_title(f'{steps_names[i]} (n={repeat})')
    ax.set_ylim([1e-6, 1e2])
    plt.legend()

# %%
r = np.sum(radiosity.form_factors==0) / radiosity.form_factors.size
print(f'ratio of zeros of form factors: {r*100:.2f}%')

# %%



