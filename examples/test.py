# %%
"""Test the radiosity.Radiosity module."""
import numpy as np
import pyfar as pf
import sparrowpy as sp
import matplotlib.pyplot as plt
from datetime import datetime
%matplotlib inline

# %%
# Define parameters
X = 5
Y = 6
Z = 4
patch_size = 1
ir_length_s = 2
sampling_rate = 1000
max_order_k = 5
speed_of_sound = 346.18
absorption = 0.1

# create geometry
walls = sp.testing.shoebox_room_stub(X, Y, Z)
source_pos = [2, 2, 2]
source = sp.geometry.SoundSource(source_pos, [0, 1, 0], [0, 0, 1])
receiver_pos = [2, 3, 2]
# create object
radiosity_fast = sp.DRadiosityFast.from_polygon(walls, patch_size)

# create directional scattering data (totally diffuse)
frequencies = np.array([500])

# set directional scattering data
coords = pf.Coordinates(0, 0, 1, weights=1)
s = pf.FrequencyData([1], frequencies)
brdf = sp.brdf.create_from_scattering(coords, coords, s)
radiosity_fast.set_wall_scattering(
    np.arange(len(walls)),
    brdf,
    sources=coords,
    receivers=coords)

# set air absorption
radiosity_fast.set_air_attenuation(
    pf.FrequencyData(
        np.zeros_like(frequencies),
        frequencies))

# set absorption coefficient
radiosity_fast.set_wall_absorption(
    np.arange(len(walls)),
    pf.FrequencyData(
        np.zeros_like(frequencies)+absorption,
        frequencies))

# calculate from factors including directivity and absorption
radiosity_fast.bake_geometry(algorithm='order')

# initialize source energy at each patch
radiosity_fast.init_source_energy(source_pos, algorithm='order')

# gather energy at receiver
radiosity_fast.calculate_energy_exchange(
    speed_of_sound=speed_of_sound,
    histogram_time_resolution=1/sampling_rate,
    histogram_length=ir_length_s,
    algorithm='order', max_depth=max_order_k)

# %%
ir_fast = radiosity_fast.collect_receiver_energy(
    receiver_pos,
    )

reverberation_fast = pf.Signal(
    ir_fast[0, 0],
    sampling_rate=sampling_rate)

# %%
S = (2*X*Y) + (2*X*Z) + (2*Y*Z)
A = S*absorption
alpha_dash = A/S
r_h = 1/4*np.sqrt(A/np.pi)
print(f'reverberation distance is {r_h}m')
V = X*Y*Z
RT = 24*np.log(10)/(speed_of_sound)*V/(-S*np.log(1-alpha_dash))
print(f'reverberation time is {RT}s')
E_reverb_analytical = 4/A
t = reverberation_fast.times
# Kuttruff Eq 4.7
w_0 = E_reverb_analytical/ V
t_0 = 0.03
# Kuttruff Eq 4.10
reverberation_analytic = w_0 * np.exp(+(
    speed_of_sound*S*np.log(1-alpha_dash)/(4*V))*(t-t_0))
reverberation_analytic = pf.Signal(
    reverberation_analytic, sampling_rate=sampling_rate)


# %%
plt.figure()
pf.plot.time(
    reverberation_analytic, dB=True, log_prefix=10,
    label=f'analytical E_rev={E_reverb_analytical:0.2f}')
# pf.plot.time(
#     reverberation_slow, dB=True, log_prefix=10,
#     label=f'simulated slow ({slow_time_s:0.2f}s)',
#     linestyle='-')
pf.plot.time(
    reverberation_fast, dB=True, log_prefix=10,
    label=f'simulated fast ({1:0.2f}s)',
    linestyle='--')

plt.legend()
plt.show()
# %%
