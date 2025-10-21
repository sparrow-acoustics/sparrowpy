# %%
import sparrowpy as sp
import numpy as np
import pyfar as pf
import sofar as sf
import matplotlib.pyplot as plt
import sys
import os
from scipy.integrate import dblquad
from numpy.polynomial.legendre import leggauss
import sparrowpy.geometry as geom


#%%
elevation = 45
elevation_rad = np.deg2rad(elevation)

distance_from_patch = 1
source = pf.Coordinates.from_spherical_elevation(0,elevation_rad,distance_from_patch)
receiver = pf.Coordinates.from_spherical_elevation(np.pi,source.elevation[0],source.radius[0])

#patch = sp.testing.shoebox_room_stub(1, 1, 1)[2]
total_distance = source.radius + receiver.radius
analytical = 1/(4*np.pi*total_distance**2)
print(f'analytical = {analytical}')
do_original_method = False
width = 2
length = 2
patch_size = 2

sampling_rate = 4000/2
speed_of_sound = 343.2
etc_duration = 0.5 # seconds
etc_time = etc_duration

patch = sp.geometry.Polygon(
        [[-width/2, -length/2, 0],
            [width/2, -length/2, 0],
            [width/2, length/2, 0],
            [-width/2, length/2, 0]],
        [1, 0, 0], [0, 0, 1])

delta_angles = 15
sampling = pf.samplings.sph_equal_angle(delta_angles)
sampling.weights = pf.samplings.calculate_sph_voronoi_weights(sampling)
sampling = sampling[sampling.z>0]
sampling.weights *= 2*np.pi

scattering_coef = [0.0]
frequencies = [300]
brdf_data = sp.brdf.create_from_scattering(sampling, sampling, pf.FrequencyData(scattering_coef, frequencies))#,pf.FrequencyData([1, 1], [300, 400]))
toplot_receiver = receiver
#######################
#####Oringinal method
if do_original_method == True:
    radi = sp.DirectionalRadiosityFast.from_polygon([patch], patch_size)
    radi.set_wall_brdf(
        np.arange(1), brdf_data, sampling, sampling)

    # set air absorption
    radi.set_air_attenuation(
        pf.FrequencyData(
            np.zeros_like(brdf_data.frequencies),
            brdf_data.frequencies))
    radi.init_source_energy(source)

    radi.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        etc_time_resolution=1/sampling_rate,
        etc_duration=etc_duration,
        max_reflection_order=0)

    etc = radi.collect_energy_receiver_mono(toplot_receiver, False)

#########Integration with BRDF#########

radi_brdf_integration = sp.DirectionalRadiosityFast.from_polygon([patch], patch_size)
radi_brdf_integration.set_wall_brdf(
    np.arange(1), brdf_data, sampling, sampling)

radi_brdf_integration._integration_method = "dblquad" # "leggauss" or "montecarlo"
radi_brdf_integration._integration_sampling = 64 #for montecarlo = N^2, for leggauss = N. leggauss max N = 30 -> already long computation. so montecarlo up to 1000 samplings on the surface
# set air absorption
radi_brdf_integration.set_air_attenuation(
    pf.FrequencyData(
        np.zeros_like(brdf_data.frequencies),
        brdf_data.frequencies))
radi_brdf_integration.init_source_energy_brdf_integration(source)
test = radi_brdf_integration._energy_exchange_etc

radi_brdf_integration.calculate_energy_exchange(
    speed_of_sound=speed_of_sound,
    etc_time_resolution=1/sampling_rate,
    etc_duration=etc_duration,
    max_reflection_order=0)

etc_brdf_integration = radi_brdf_integration.collect_energy_receiver_mono_brdf_integration(toplot_receiver, False)

if do_original_method == True:
    Numerical = np.sum(etc.time)
Numerical_brdf_integration = np.sum(etc_brdf_integration.time)


c = Numerical_brdf_integration