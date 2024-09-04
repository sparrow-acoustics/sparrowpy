# %%
import os

import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
import sparapy as sp
import sparapy.geometry as geo
import sparapy.radiosity as radiosity
from sparapy.sound_object import Receiver, SoundSource
import matplotlib.pyplot as plt
# %%

for patch_size in [1, 1/3, 1/5]:
    for sh_order in [1, 3, 5]:
        brdf_s_0 = "brdf_tmp.sofa"
        coords = pf.samplings.sph_gaussian(sh_order=sh_order)
        coords = coords[coords.z > 0]
        sp.brdf.create_from_scattering(
        brdf_s_0, coords, coords, pf.FrequencyData(0, [100]))
        max_order_k = 2
        X = 1
        Y = 1
        x_s = 10
        y_s = 0
        z_s = 10
        x_r = -10
        y_r = 0
        z_r = 10
        ir_length_s = 0.15
        sampling_rate = 50

        # create geometry
        ground = geo.Polygon(
            [[-X/2, -Y/2, 0], [X/2, -Y/2, 0], [X/2, Y/2, 0], [-X/2, Y/2, 0]],
            [1, 0, 0], [0, 0, 1])
        source = SoundSource([x_s, y_s, z_s], [0, 1, 0], [0, 0, 1])

        # new approach
        radi = radiosity.DirectionalRadiosity(
            [ground], patch_size, max_order_k, ir_length_s,
            brdf_s_0, speed_of_sound=343, sampling_rate=sampling_rate)

        radi.run(source)
        ir = radi.energy_at_receiver(
            Receiver([x_r, y_r, z_r], [0, 1, 0], [0, 0, 1]))

        ir_desired = np.zeros_like(ir)
        # add direct sound
        direct_distance = np.linalg.norm([x_r - x_s, y_r - y_s, z_r - z_s])
        direct_samples = int(direct_distance/343*sampling_rate)
        ir_desired[:, direct_samples] = 1/(4*np.pi*direct_distance**2)
        # add reflection sound
        reflection_distance = np.linalg.norm([x_r - x_s, y_r - y_s, -z_r - z_s])
        reflection_samples = int(reflection_distance/343*sampling_rate)
        ir_desired[:, reflection_samples] = 1/(4*np.pi*reflection_distance**2)

        # npt.assert_almost_equal(10*np.log10(ir), 10*np.log10(ir_desired))
        print(f'Patch size: {patch_size:.1f}, order: {sh_order} -> {10*np.log10(ir[0, 4]):.2f}dB soll -> {10*np.log10(ir_desired[0, 4]):.2f}dB')
        ir[0, 4]/ir_desired[0, 4]

# %%

for patch_size in [1, 1/3, 1/5]:
    for sh_order in [1, 3, 5]:
        brdf_s_0 = "brdf_tmp.sofa"
        coords = pf.samplings.sph_gaussian(sh_order=sh_order)
        coords = coords[coords.z > 0]
        sp.brdf.create_from_scattering(
        brdf_s_0, coords, coords, pf.FrequencyData(0, [100]))
        max_order_k = 2
        x_s = 0
        y_s = 0
        z_s = 0
        x_r = 3
        y_r = 0
        z_r = 1
        ir_length_s = 0.25
        sampling_rate = 500

        # create geometry
        ground = geo.Polygon(
            [[0.5, 0.5, 1], [0.5, -0.5, 1], [1.5, -0.5, 1], [1.5, 0.5, 1]],
            [1, 0, 0], [0, 0, 1])
        ground2 = geo.Polygon(
            [[1.5, 0.5, 0], [1.5, -0.5, 0], [2.5, -0.5, 0], [2.5, 0.5, 0]],
            [1, 0, 0], [0, 0, 1])
        source = SoundSource([x_s, y_s, z_s], [0, 1, 0], [0, 0, 1])

        # new approach
        radi = radiosity.DirectionalRadiosity(
            [ground, ground2], patch_size, max_order_k, ir_length_s,
            brdf_s_0, speed_of_sound=343, sampling_rate=sampling_rate)

        radi.run(source)
        ir = radi.energy_at_receiver(
            Receiver([x_r, y_r, z_r], [0, 1, 0], [0, 0, 1]))

        ir_desired = np.zeros_like(ir)
        # add direct sound
        direct_distance = np.linalg.norm([x_r - x_s, y_r - y_s, z_r - z_s])
        direct_samples = int(direct_distance/343*sampling_rate)
        ir_desired[:, direct_samples] = 1/(4*np.pi*direct_distance**2)
        # add reflection sound
        reflection_distance = np.linalg.norm([x_r - 0, y_r - 0, z_r +2])
        reflection_samples = int(reflection_distance/343*sampling_rate)
        ir_desired[:, reflection_samples] = 1/(4*np.pi*reflection_distance**2)

        # npt.assert_almost_equal(10*np.log10(ir), 10*np.log10(ir_desired))
        print(f'direct: Patch: {patch_size:.1f}, order: {sh_order} -> {10*np.log10(ir[0, direct_samples]):.2f}dB soll -> {10*np.log10(ir_desired[0, direct_samples]):.2f}dB')
        print(f'refl: Patch: {patch_size:.1f}, order: {sh_order} -> {10*np.log10(ir[0, reflection_samples]):.2f}dB soll -> {10*np.log10(ir_desired[0, reflection_samples]):.2f}dB')
        ir[0, 4]/ir_desired[0, 4]

# %%
max_order_k = 2
x_s = 0
y_s = 0
z_s = 0
x_r = 3
y_r = 0
z_r = 1
ir_length_s = 0.25
sampling_rate = 500

# create geometry
ground = geo.Polygon(
    [[0.5, 0.5, 1], [0.5, -0.5, 1], [1.5, -0.5, 1], [1.5, 0.5, 1]],
    [1, 0, 0], [0, 0, 1])
ground2 = geo.Polygon(
    [[1.5, 0.5, 0], [1.5, -0.5, 0], [2.5, -0.5, 0], [2.5, 0.5, 0]],
    [1, 0, 0], [0, 0, 1])
source = SoundSource([x_s, y_s, z_s], [0, 1, 0], [0, 0, 1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for patch in [ground, ground2]:
    patch.plot(ax=ax)
ax.scatter(x_s, y_s, z_s, c='b')
ax.scatter(x_r, y_r, z_r, c='r')
# %%
