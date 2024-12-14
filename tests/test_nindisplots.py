# %%
import numpy as np
import trimesh
from sparapy import geometry as geo
import matplotlib.pyplot as plt
import pyfar as pf
import sparapy as sp


# %%
# geometry

X = 5
Y = 6
Z = 4
patch_size = 0.5
ir_length_s = 2
sampling_rate = 1000
max_order_k = 3
speed_of_sound = 343

absorption = 0.1
S = (2*X*Y) + (2*X*Z) + (2*Y*Z)
A = S*absorption
alpha_dash = A/S
r_h = 1/4*np.sqrt(A/np.pi)
print(f'reverberation distance is {r_h}m')
V = X*Y*Z
RT = 24*np.log(10)/(speed_of_sound)*V/(-S*np.log(1-alpha_dash))
print(f'reverberation time is {RT}s')
# create geometry
walls = sp.testing.shoebox_room_stub(X, Y, Z)
source = sp.geometry.SoundSource([2, 2, 2], [0, 1, 0], [0, 0, 1])

## new approach
radi = sp.radiosity.Radiosity(
    walls, patch_size, max_order_k, ir_length_s,
    speed_of_sound=speed_of_sound, sampling_rate=sampling_rate)

radi.run(source)

# %%
for wall in radi.patch_list:
    wall.plot_energy_patches_time(100)

# %%
