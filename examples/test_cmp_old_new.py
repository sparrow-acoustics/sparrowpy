# %%
import numpy as np
import numpy.testing as npt
import pytest
import os
import pyfar as pf
import matplotlib.pyplot as plt

import sparapy as sp
%matplotlib ipympl
#%%
patch_size = 1
max_order_k = 2
length_histogram = 0.2
time_resolution = 1e-3
speed_of_sound = 346.18
walls = sp.testing.shoebox_room_stub(5, 6, 4)
gaussian = pf.samplings.sph_gaussian(sh_order=1)
gaussian = gaussian[gaussian.z>0]
sources = gaussian.copy()
receivers = gaussian.copy()
frequencies = pf.dsp.filter.fractional_octave_frequencies(
    1, (100, 1000))[0]
data = np.ones((sources.csize, receivers.csize, frequencies.size))
data = pf.FrequencyData(data, frequencies)

radiosity_old = sp.radiosity.Radiosity(
    walls, patch_size, max_order_k, length_histogram,
    speed_of_sound=speed_of_sound,
    sampling_rate=1/time_resolution, absorption=0)

source_pos = np.array([2, 2, 2])
receiver_pos = np.array([3, 4, 2])
radiosity_old.run(
    sp.geometry.SoundSource(source_pos, [1, 0, 0], [0, 0, 1]))
histogram_old = radiosity_old.energy_at_receiver(
    sp.geometry.Receiver(receiver_pos, [1, 0, 0], [0, 0, 1]), ignore_direct=True)
histogram_old_1 = radiosity_old.energy_at_receiver(
    sp.geometry.Receiver(receiver_pos, [1, 0, 0], [0, 0, 1]), ignore_direct=True, max_order_k=1)
# histogram_old = histogram_old-histogram_old_1
radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(
    walls, patch_size)

radiosity.set_wall_scattering(
    np.arange(len(walls)), data, sources, receivers)
radiosity.set_air_attenuation(
    pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
radiosity.set_wall_absorption(
    np.arange(len(walls)),
    pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
radiosity.check_visibility()
radiosity.calculate_form_factors()
radiosity.calculate_form_factors_directivity()

radiosity.init_energy_recursive(source_pos)
histogram = radiosity.calculate_energy_exchange_recursive(
    receiver_pos, speed_of_sound, time_resolution, length_histogram,
    threshold=0, max_time=np.inf, max_depth=max_order_k)
histogram1 = radiosity.calculate_energy_exchange_recursive(
    receiver_pos, speed_of_sound, time_resolution, length_histogram,
    threshold=0, max_time=np.inf, max_depth=max_order_k-1)
histogram_diff = histogram - histogram1
#%%
# compare histogram
histogram_diff_sig = pf.Signal(histogram_diff, 1/time_resolution)
histogram_sig = pf.Signal(histogram, 1/time_resolution)
histogram_old_sig = pf.Signal(histogram_old, 1/time_resolution)
histogram_old_1_sig = pf.Signal(histogram_old_1, 1/time_resolution)
histogram_old_diff_sig = histogram_old_sig-histogram_old_1_sig

merged = pf.utils.concatenate_channels([
    histogram_old_diff_sig[0],
    # histogram_old_1_sig[0],
    # histogram_old_sig[0],
    histogram_diff_sig,
    ])
plt.figure()
pf.plot.time(merged, dB=True, log_prefix=10)
# pf.plot.time(histogram_old_sig, dB=True, log_prefix=10)



# %%
# compare histogram
for i in range(4):
    assert np.sum(histogram[i, :])>0
    npt.assert_allclose(
        np.sum(histogram_old_diff_sig.time[i, :]), np.sum(histogram_diff[0, :]),
        err_msg=f'histogram i_bin={i}')

# %%
