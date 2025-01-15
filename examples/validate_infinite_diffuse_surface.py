# %%
import numpy as np
import sparrowpy as sp
import pyfar as pf

# %%
# Setup a simulation for infinite plane
width = 10
depth = 10
patch_size = 1

plane = sp.geometry.Polygon(
    [[-width/2, -depth/2, 0],
     [-width/2, depth/2, 0],
     [width/2, depth/2, 0],
     [width/2, depth/2, 0]],
    up_vector=np.array([1.,0.,0.]),
    normal=np.array([0.,0.,1.]))

#simulation parameters
speed_of_sound = 346.18

## PREPARE RADIOSITY SIMULATION ##
#initialize radiosity class instance based on plane "radi"
radi = sp.DRadiosityFast.from_polygon([plane], patch_size)

#set scattering distribution (lambertian surface)
coords = pf.samplings.sph_gaussian(sh_order=31)
coords = coords[coords.z>0]
s = pf.FrequencyData([1], 100)
brdf = sp.brdf.create_from_scattering(coords, coords, s)
radi.set_wall_scattering(np.arange(1), brdf,
                            sources=coords,
                            receivers=coords)

#set atmospheric attenuation and surface absorption coefficient to 0
radi.set_air_attenuation(
                pf.FrequencyData(
                    np.zeros_like(s.frequencies),
                    s.frequencies))
radi.set_wall_absorption(
                np.arange(1),
                pf.FrequencyData(
                    np.zeros_like(s.frequencies),
                    s.frequencies))

## RUN SIMULATION ##
#precompute energy relationships
radi.bake_geometry()

# %%
# Test case one
source = pf.Coordinates(0, 0, 3)
receiver = pf.Coordinates(0, 0, 3)
sampling_rate = 10

max_sound_path_length = receiver.radius[0] + source.radius[0]
max_histogram_length = max_sound_path_length/speed_of_sound
#initialize source energy (source cast)
radi.init_source_energy(source_position=source.cartesian)

#energy propagation simulation
radi.calculate_energy_exchange(
    speed_of_sound=speed_of_sound,
    histogram_time_resolution=1/sampling_rate,
    histogram_length=1.*max_histogram_length,
    max_depth=5,
)
histogram_diffuse = radi.collect_receiver_energy(
    receiver_pos=receiver.cartesian,
    speed_of_sound=speed_of_sound,
    histogram_time_resolution=1/sampling_rate,
    )
print(histogram_diffuse) 
# %%
