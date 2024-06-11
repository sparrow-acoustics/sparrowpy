from sparapy.radiosity_fast import DRadiosityFast
import sparapy.geometry as geo
import sparapy as sp
import cProfile
import pyfar as pf
import sofar as sf
import os
import tqdm
import time
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

# def calc_form_factor(rad):
#     if len(rad.patch_list) > 1:
#         for patches in rad.patch_list:
#             patches.calculate_form_factor(rad.patch_list)

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
repeat = 2
steps_names = [
    'create patches', 'init energy', 'form factor',
    'energy exchange', 'collect energy',
    ]
steps = len(steps_names)
fast_second = np.zeros((steps, n_max, repeat))
memory_fast = np.zeros((steps, n_max, repeat))
memory_slow = np.zeros((steps, n_max, repeat))
# memory_slow[:] = np.nan
slow = np.zeros((steps, n_max, repeat))
# slow[:] = np.nan
number_of_patches = np.zeros((n_max))

# run one time, to get complied.
radiosity = DRadiosityFast.from_polygon(sample_walls, 1)

# radiosity.set_wall_scattering(
#     np.arange(6), data, sources, receivers)
# radiosity.set_air_attenuation(
#     pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
# radiosity.set_wall_absorption(
#     np.arange(6),
#     pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
radiosity.check_visibility()

t0 = time.time()
radiosity.calculate_form_factors(method='kang')
tkang = time.time()-t0

kang = radiosity.form_factors

t0 = time.time()
radiosity.calculate_form_factors(method='universal')
tuniv = time.time()-t0

univ = radiosity.form_factors

msr_error = np.mean(np.sqrt(np.square(kang-univ)))

print("\\n\n\nRUNTIME:")
print(f"kang {tkang: .10f}s")
print(f"univ: {tuniv: .10f}s")

print("\n\nMSR: ")
print(f"abs: {msr_error: .10f}")
print(f"rel: {100*msr_error/np.mean(kang): .10f}%")
print("end")


