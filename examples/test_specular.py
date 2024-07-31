"""Test specular reflections."""
# %%
import numpy as np
import sparapy as sp
import pyfar as pf
import os
import matplotlib.pyplot as plt
# %matplotlib ipympl

# %%
sample_walls = sp.testing.shoebox_room_stub(1, 1, 1)
walls = [0, 1]
patch_size = 0.2
# %%
data, sources, receivers = pf.io.read_sofa(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'specular.s_gaussian_19.sofa'))

data = pf.FrequencyData(np.concatenate(
    (data.freq, data.freq), axis=-1),
    [data.frequencies[0], data.frequencies[0]*2])

# %%
source_pos = np.array([0.5, 0.5, 0.5])
receiver_pos = np.array([0.5, 0.5, 0.5])
wall_source = sample_walls[walls[0]]
wall_receiver = sample_walls[walls[1]]
walls = [wall_source, wall_receiver]
length_histogram = 0.1
time_resolution = 1e-4
k = 10
speed_of_sound = 346.18

# use DirectionDirectivity instead
radiosity_old = sp.radiosity.Radiosity(
    walls, patch_size, k, length_histogram,
    speed_of_sound=speed_of_sound,
    sampling_rate=1/time_resolution)
radiosity_old.run(
    sp.geometry.SoundSource(source_pos, [1, 0, 0], [0, 0, 1]))
histogram_old = radiosity_old.energy_at_receiver(
    sp.geometry.Receiver(receiver_pos, [1, 0, 0], [0, 0, 1]),
    ignore_direct=True)

# %%
old_sig = pf.Signal(histogram_old, 1/time_resolution)

soll = pf.Signal(np.zeros_like(histogram_old), 1/time_resolution)
# 1st order
delta_d = 1
soll.time[:, int(delta_d/speed_of_sound/time_resolution)] = 1 / (4*np.pi*delta_d**2)
# 2nd order
delta_d = 2
soll.time[:, int(delta_d/speed_of_sound/time_resolution)] = 1 / (4*np.pi*delta_d**2)
# 3rd order
delta_d = 1 # todo
soll.time[:, int(delta_d/speed_of_sound/time_resolution)] = 1 / (4*np.pi*delta_d**2)

# %% this was a copy from chatgpt
