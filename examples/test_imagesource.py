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
# try loading and reading using trimesh 

list_polygon = read_geometry.read_geometry('models/shoebox_3.stl')

# %%
# trimesh radiosity using normal patches (non directional)

walls = list_polygon
patch_size = 1
source_pos = np.array([0.5, 0.5, 0.5])
receiver_pos = np.array([0.25, 0.25, 0.25])

length_histogram = 0.1
time_resolution = 1e-4
k = 5
speed_of_sound = 346.18

path_sofa = os.path.join(
    os.path.dirname(__file__), 'test_brdf_19.sofa')

# use DirectionDirectivity instead
radiosity_old = sp.radiosity.Radiosity(
    walls, patch_size, k, length_histogram,
    speed_of_sound=speed_of_sound,
    sampling_rate=1/time_resolution)
radiosity_old.run(
    sp.geometry.SoundSource(source_pos, [1, 0, 0], [0, 0, 1]))

# problem starts here don't know why 
histogram_old = radiosity_old.energy_at_receiver(
    sp.geometry.Receiver(receiver_pos, [1, 0, 0], [0, 0, 1])) 

pf.plot.use()
plt.figure()
radiosity_trimesh = pf.Signal(histogram_old, 10000)
ax = pf.plot.time(radiosity_trimesh,dB=True)
ax.set_xlim((0,0.04))
plt.show()

# %%
# normal radiosity, normal polygons

walls = sp.testing.shoebox_room_stub(3, 3, 3)
patch_size = 1
source_pos = np.array([0.5, 0.5, 0.5])
receiver_pos = np.array([0.25, 0.25, 0.25])

length_histogram = 0.1
time_resolution = 1e-4
k = 5
speed_of_sound = 346.18

path_sofa = os.path.join(
    os.path.dirname(__file__), 'test_brdf_19.sofa')

# use DirectionDirectivity instead
radiosity_old = sp.radiosity.Radiosity(
    walls, patch_size, k, length_histogram,
    speed_of_sound=speed_of_sound,
    sampling_rate=1/time_resolution)
radiosity_old.run(
    sp.geometry.SoundSource(source_pos, [1, 0, 0], [0, 0, 1]))

# problem starts here don't know why 
histogram_old = radiosity_old.energy_at_receiver(
    sp.geometry.Receiver(receiver_pos, [1, 0, 0], [0, 0, 1])) 

pf.plot.use()
plt.figure()
radiosity_trimesh = pf.Signal(histogram_old, 10000)
ax = pf.plot.time(radiosity_trimesh,dB=True)
ax.set_xlim((0,0.04))
plt.show()

# %%
# trimesh dir rad using brdf

walls = list_polygon
patch_size = 0.2
source_pos = np.array([0.5, 0.5, 0.5])
receiver_pos = np.array([0.25, 0.25, 0.25])

length_histogram = 0.1
time_resolution = 1e-4
k = 5
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

pf.plot.use()
plt.figure()
radiosity_trimesh = pf.Signal(histogram_old, 10000)
ax = pf.plot.time(radiosity_trimesh,dB=True)
ax.set_xlim((0,0.01))
plt.show()


# %%
# image source method
RoomSizes = (3, 3, 3) 
WallsHesse = ims.get_walls_hesse(*RoomSizes)

WallsR = 0.9
SourcePos = [0.50, 0.50, 0.50]
MaxOrder = 10

ISList_valid = ims.calculate_image_sources(WallsHesse, SourcePos, MaxOrder)

ReceiverPos = [0.25, 0.25, 0.25]
ISList_audible = ims.filter_image_sources(ISList_valid, WallsHesse, ReceiverPos)

impulse_response = ims.calculate_impulse_response(ISList_audible, WallsR, ReceiverPos)
energy = np.square(impulse_response)
signal = pf.Signal(energy, 10000)

pf.plot.use()
plt.figure()
ax = pf.plot.time(signal,dB=True)
ax.set_xlim((0,0.01))
plt.show()

# %%
# normal geometry dir rad using brdf

walls = sp.testing.shoebox_room_stub(3, 3, 3)
patch_size = 0.2
source_pos = np.array([0.5, 0.5, 0.5])
receiver_pos = np.array([0.25, 0.25, 0.25])

length_histogram = 0.1
time_resolution = 1e-4
k = 5
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

pf.plot.use()
plt.figure()
radiosity1 = pf.Signal(histogram_old, 10000)
ax = pf.plot.time(radiosity1,dB=True)
ax.set_xlim((0,0.01))
plt.show()



# %%
# radiosity 1

walls = sp.testing.shoebox_room_stub(3, 3, 3)
patch_size = 0.2
source_pos = np.array([0.5, 0.5, 0.5])
receiver_pos = np.array([0.25, 0.25, 0.25])

length_histogram = 0.1
time_resolution = 1e-3 #before this was -4
k = 5
speed_of_sound = 346.18

path_sofa = os.path.join(
    os.path.dirname(__file__), 'specular.s_gaussian_1.sofa')

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

pf.plot.use()
plt.figure()
radiosity1 = pf.Signal(histogram_old, 1000)
ax = pf.plot.time(radiosity1,dB=True)
ax.set_xlim((0,0.01))
plt.show()

# %%
# radiosity 11

walls = sp.testing.shoebox_room_stub(1, 1, 1)
patch_size = 0.2
source_pos = np.array([0.5, 0.5, 0.5])
receiver_pos = np.array([0.25, 0.25, 0.25])

length_histogram = 0.1
time_resolution = 1e-4
k = 5
speed_of_sound = 346.18

path_sofa = os.path.join(
    os.path.dirname(__file__), 'specular.s_gaussian_11.sofa')

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

pf.plot.use()
plt.figure()
radiosity11 = pf.Signal(histogram_old, 10000)
ax = pf.plot.time(radiosity11,dB=True)
ax.set_xlim((0,0.01))
plt.show()

# %%
# radiosity 19

walls = sp.testing.shoebox_room_stub(1, 1, 1)
patch_size = 0.2
source_pos = np.array([0.5, 0.5, 0.5])
receiver_pos = np.array([0.25, 0.25, 0.25])

length_histogram = 0.1
time_resolution = 1e-4
k = 5
speed_of_sound = 346.18

path_sofa = os.path.join(
    os.path.dirname(__file__), 'specular.s_gaussian_19.sofa')

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

pf.plot.use()
plt.figure()
radiosity19 = pf.Signal(histogram_old, 10000)
ax = pf.plot.time(radiosity19,dB=True)
ax.set_xlim((0,0.01))
plt.show()

# %% 
# comparison directivities

pf.plot.use()
plt.figure()
ax = pf.plot.time(radiosity1, dB=True, label="Gaussian 1")
ax = pf.plot.time(radiosity11, dB=True, label="Gaussian 11")
ax = pf.plot.time(radiosity19, dB=True, label="Gaussian 19")

ax.set_xlim((0,0.02))

ax.legend(loc="upper right")
plt.legend(title='Radiosity Directivities for k=5')
plt.show()

# %% 
# comparison image source vs radiosity
impulse_response = ims.calculate_impulse_response(ISList_audible, WallsR, ReceiverPos)
energy = np.square(impulse_response)
image_source = pf.Signal(energy, 10000)
pf.plot.use()
plt.figure()
ax = pf.plot.time(radiosity, dB=True, label="Radiosity")
ax = pf.plot.time(image_source, dB=True, label="Image Source Method")
ax.set_xlim((0,0.02))

ax.legend(loc="upper right")
plt.legend(title='Radiosity vs Image Source Method')
plt.show()

# %%
# normal radiosity without directivities

walls = sp.testing.shoebox_room_stub(1, 1, 1)
patch_size = 0.2
source_pos = np.array([0.5, 0.5, 0.5])
receiver_pos = np.array([0.25, 0.25, 0.25])

length_histogram = 0.1
time_resolution = 1e-4
k = 5
speed_of_sound = 346.18

# use DirectionDirectivity instead
radiosity_old = sp.radiosity.Radiosity(
    walls, patch_size, k, length_histogram,
    speed_of_sound=speed_of_sound,
    sampling_rate=1/time_resolution)
radiosity_old.run(
    sp.geometry.SoundSource(source_pos, [1, 0, 0], [0, 0, 1]))

# problem starts here don't know why 
histogram_old = radiosity_old.energy_at_receiver(
    sp.geometry.Receiver(receiver_pos, [1, 0, 0], [0, 0, 1]),k,ignore_direct=False) 

pf.plot.use()
plt.figure()
radiosity = pf.Signal(histogram_old, 10000)
ax = pf.plot.time(radiosity,dB=True)
ax.set_xlim((0,0.01))
plt.show()

# %%
# compare normal vs directivity radiosity 

pf.plot.use()
plt.figure()
ax = pf.plot.time(radiosity, dB=True, label="No Directivity")
ax = pf.plot.time(radiosity1, dB=True, label="Gaussian 1")
ax = pf.plot.time(radiosity11, dB=True, label="Gaussian 11")
ax = pf.plot.time(radiosity19, dB=True, label="Gaussian 19")

ax.set_xlim((0,0.02))

ax.legend(loc="upper right")
plt.legend(title='Radiosity Directivities for k=5')
plt.show()
# %%


