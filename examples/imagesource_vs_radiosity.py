# %%
# import modules
import numpy as np
import sparapy as sp
from sparapy import image_source as ims

import pyfar as pf
import matplotlib.pyplot as plt
%matplotlib ipympl

import os
from sparapy import read_geometry
import trimesh


# %%
# directional radiosity 

k = 10
patch_size = 1

walls = sp.testing.shoebox_room_stub(3, 3, 3)
source_pos = np.array([0.5, 0.5, 0.5])
receiver_pos = np.array([0.25, 0.25, 0.25])

length_histogram = 0.1
time_resolution = 1e-4

speed_of_sound = 346.18

path_sofa = os.path.join(
    os.path.dirname(__file__), 'test_brdf_19.sofa')

# use DirectionDirectivity instead
radiosity_old = sp.radiosity.DirectionalRadiosity(
    walls, patch_size, k, length_histogram,path_sofa,
    speed_of_sound=speed_of_sound,
    sampling_rate=1/time_resolution)
radiosity_old.run(
    sp.geometry.SoundSource(source_pos, [1, 0, 0], [0, 0, 1]))

# problem starts here don't know why 
histogram_old = radiosity_old.energy_at_receiver(
    sp.geometry.Receiver(receiver_pos, [1, 0, 0], [0, 0, 1])) 

radiosity = pf.Signal(histogram_old, 10000)


# pf.plot.use()
# plt.figure()

# ax = pf.plot.time(radiosity,dB=True)
# ax.set_xlim((0,0.01))
# plt.show()


# %%
# image source method

MaxOrder = 5

RoomSizes = (3, 3, 3) 
WallsHesse = ims.get_walls_hesse(*RoomSizes)

WallsR = 0.9
SourcePos = [0.50, 0.50, 0.50]


ISList_valid = ims.calculate_image_sources(WallsHesse, SourcePos, MaxOrder)

ReceiverPos = [0.25, 0.25, 0.25]
ISList_audible = ims.filter_image_sources(ISList_valid, WallsHesse, ReceiverPos)

impulse_response = ims.calculate_impulse_response(ISList_audible, WallsR, ReceiverPos)
energy = np.square(impulse_response)
image_source = pf.Signal(energy, 10000)

# pf.plot.use()
# plt.figure()
# ax = pf.plot.time(image_source,dB=True)
# ax.set_xlim((0,0.01))
# plt.show()


# %% 
# comparison image source vs radiosity

pf.plot.use()
plt.figure()
ax = pf.plot.time(radiosity, dB=True, label="Radiosity")
ax = pf.plot.time(image_source, dB=True, label="Image Source Method")
#ax.set_xlim((0,0.02))


title = title = f"Radiosity Order {k} Patch Size {patch_size} vs Image Source Order {MaxOrder}"
ax.legend(loc="upper right")
plt.title(title)
plt.show()


# %%
