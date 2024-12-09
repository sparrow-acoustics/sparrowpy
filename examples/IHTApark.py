# %%
# imports

import numpy as np
import trimesh
from sparapy import geometry as geo
import matplotlib.pyplot as plt
import pyfar as pf
from sparapy import io
from sparapy import brdf
import sparapy as sp

%matplotlib ipympl

# %%
# imports

mesh = trimesh.load_mesh('IHTApark_acoustic_geo_cloth.stl')
mesh.show()

# %%
path = 'IHTApark_acoustic_geo_cloth.stl'
walls = io.read_geometry(path,'triangle')

plt.figure()
pf.plot.use()
ax = plt.axes(projection='3d')

# Loop through each face in list_polygon and plot it
for face in walls:
    face.plot(ax)  # Assuming plot method accepts ax and color

# Show the plot
plt.show()
# %%
