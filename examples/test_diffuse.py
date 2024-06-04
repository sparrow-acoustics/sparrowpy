# %%
import numpy as np
import sparapy as sp
import pyfar as pf
import os
%matplotlib ipympl

# %%
sample_walls = sp.testing.shoebox_room_stub(1, 1, 1)
walls = [0, 1]
patch_size = 0.5

#%%
data, sources, receivers = pf.io.read_sofa(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'specular.s_gaussian_19.sofa'))
data.freq = np.ones_like(data.freq)

#%%
source_pos = np.array([0.5, 0.5, 0.5])
receiver_pos = np.array([0.5, 0.5, 0.5])
wall_source = sample_walls[walls[0]]
wall_receiver = sample_walls[walls[1]]
walls = [wall_source, wall_receiver]
length_histogram = 0.1
time_resolution = 1e-3
k = 5
speed_of_sound = 346.18

radiosity_old = sp.radiosity.Radiosity(
    walls, patch_size, k, length_histogram,
    speed_of_sound=speed_of_sound,
    sampling_rate=1/time_resolution, absorption=0)
radiosity_old.run(
    sp.geometry.SoundSource(source_pos, [1, 0, 0], [0, 0, 1]))
histogram_old = radiosity_old.energy_at_receiver(
    sp.geometry.Receiver(receiver_pos, [1, 0, 0], [0, 0, 1]), ignore_direct=True)

radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(
    walls, patch_size)

#%%
radiosity.set_wall_scattering(
    np.arange(len(walls)), data, sources, receivers)
radiosity.set_air_attenuation(
    pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
radiosity.set_wall_absorption(
    np.arange(len(walls)),
    pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
#%%
radiosity.check_visibility()
radiosity.calculate_form_factors()
radiosity.calculate_form_factors_directivity()

# %%

radiosity.calculate_energy_exchange(k)

# %%
radiosity.init_energy(source_pos)
histogram = radiosity.collect_energy_receiver(
    receiver_pos, speed_of_sound=speed_of_sound,
    histogram_time_resolution=time_resolution,
    histogram_time_length=length_histogram)
#%%
old_sig = pf.Signal(histogram_old, 1/time_resolution)
# old_sig.freq[:, 0] =0
new_sig = pf.Signal(histogram.T[0, :], 1/time_resolution) /1
import matplotlib.pyplot as plt

soll = pf.Signal(np.zeros_like(histogram_old), 1/time_resolution)
radiosity.patches_center - source_pos
result = pf.utils.concatenate_channels([ old_sig, new_sig])

plt.figure()
# pf.plot.time(pf.signals.impulse(length_histogram/time_resolution,int(1/speed_of_sound/time_resolution), sampling_rate=1/time_resolution), dB=True, log_prefix=10)
# pf.plot.time(pf.signals.impulse(length_histogram/time_resolution,int(2/speed_of_sound/time_resolution), sampling_rate=1/time_resolution), dB=True, log_prefix=10)
ax=pf.plot.time(result, dB=True, log_prefix=10)
ax.set_xlim((0, 0.04))
ax.set_ylim((-100, 0))
plt.legend(['soll', 'old', 'new'])
plt.show()

print(f'sum {np.sum(result.time, axis=-1)}')
e = np.sum(result.time, axis=-1)
print(f'sum_ratio {e[0]/e[1]}')

# %%
