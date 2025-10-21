# %%

import sparrowpy as sp
import numpy as np
import pyfar as pf
import sofar as sf
import matplotlib.pyplot as plt
from tqdm import tqdm


# %%
# Define parameters
patch_size = 2
etc_duration = 1
etc_time_resolution = 1/2
max_reflection_order = 1
speed_of_sound = 343.2
absorption = 0

width = patch_size
length = patch_size
# %%

plane = sp.geometry.Polygon(
        [[-width/2, -length/2, 0],
            [width/2, -length/2, 0],
            [width/2, length/2, 0],
            [-width/2, length/2, 0]],
        [1, 0, 0], [0, 0, 1])

# %%
# create geometry
source = pf.Coordinates(-1, 0, 1)
receiver = pf.Coordinates(1, 0, 1)

radiosity_fast = sp.DirectionalRadiosityFast.from_polygon(
    [plane], patch_size)
brdf_sources = pf.Coordinates(0, 0, 1, weights=1)
brdf_receivers = pf.Coordinates(0, 0, 1, weights=1)
frequencies = np.array([1000])
brdf = sp.brdf.create_from_scattering(
    brdf_sources,
    brdf_receivers,
    pf.FrequencyData(1, frequencies),
    pf.FrequencyData(absorption, frequencies))

# set directional scattering data
radiosity_fast.set_wall_brdf(
    np.arange(1), brdf, brdf_sources, brdf_receivers)
# set air absorption
radiosity_fast.set_air_attenuation(
    pf.FrequencyData(
        np.zeros_like(brdf.frequencies),
        brdf.frequencies))

radiosity_fast.init_source_energy(source)

# gather energy at receiver
radiosity_fast.calculate_energy_exchange(
    speed_of_sound=speed_of_sound,
    etc_time_resolution=etc_time_resolution,
    etc_duration=etc_duration,
    max_reflection_order=0)
etc_radiosity_diff = radiosity_fast.collect_energy_receiver_mono(
    receivers=receiver)



# %%
# create directional scattering data (totally diffuse)
energy_radiosity = []
angles_list = [10, 15, 30]
for angles in angles_list:
    radiosity_fast = sp.DirectionalRadiosityFast.from_polygon(
        [plane], patch_size)
    brdf_sources = pf.samplings.sph_equal_angle(angles)
    brdf_sources.weights = pf.samplings.calculate_sph_voronoi_weights(
        brdf_sources)
    brdf_sources = brdf_sources[brdf_sources.z>0]
    brdf_receivers = pf.samplings.sph_equal_angle(angles)
    brdf_receivers.weights = pf.samplings.calculate_sph_voronoi_weights(
        brdf_receivers)
    brdf_receivers = brdf_receivers[brdf_receivers.z>0]
    frequencies = np.array([1000])
    brdf = sp.brdf.create_from_scattering(
        brdf_sources,
        brdf_receivers,
        pf.FrequencyData(1, frequencies),
        pf.FrequencyData(absorption, frequencies))

    # set directional scattering data
    radiosity_fast.set_wall_brdf(
        np.arange(1), brdf, brdf_sources, brdf_receivers)
    # set air absorption
    radiosity_fast.set_air_attenuation(
        pf.FrequencyData(
            np.zeros_like(brdf.frequencies),
            brdf.frequencies))

    radiosity_fast._integration_method = "montecarlo"
    n_integrations = [100, 200, 500, 1000, 2000]
    energy_radiosity_loc = []
    for n_integration in tqdm(n_integrations):
        radiosity_fast._integration_sampling = n_integration
        radiosity_fast.init_source_energy_brdf_integration(source)

        # gather energy at receiver
        radiosity_fast.calculate_energy_exchange(
            speed_of_sound=speed_of_sound,
            etc_time_resolution=etc_time_resolution,
            etc_duration=etc_duration,
            max_reflection_order=0)
        etc_radiosity = radiosity_fast.collect_energy_receiver_mono_brdf_integration(
            receivers=receiver)

        energy_radiosity_loc.append(np.sum(etc_radiosity.time))
    energy_radiosity_loc = np.array(energy_radiosity_loc)
    energy_radiosity.append(energy_radiosity_loc)
energy_radiosity = np.array(energy_radiosity)

# %%
# radiosity_fast._integration_method = "leggaus"
# energy_radiosity_leg = []
# n_integrations_leg = [4, 8, 16]
# for n_integration in tqdm(n_integrations_leg):
#     radiosity_fast._integration_sampling = n_integration
#     radiosity_fast.init_source_energy_brdf_integration(source)

#     # gather energy at receiver
#     radiosity_fast.calculate_energy_exchange(
#         speed_of_sound=speed_of_sound,
#         etc_time_resolution=etc_time_resolution,
#         etc_duration=etc_duration,
#         max_reflection_order=0)
#     etc_radiosity = radiosity_fast.collect_energy_receiver_mono_brdf_integration(
#         receivers=receiver)

#     energy_radiosity_leg.append(np.sum(etc_radiosity.time))
# energy_radiosity_leg = np.array(energy_radiosity_leg)

# %%

etc_radiosity_diff_ = np.sum(etc_radiosity_diff.time)

# %%
plt.figure()
plt.plot(n_integrations, energy_radiosity.T, 'o-', label=[f'{a}°'for a in angles_list])
plt.hlines(etc_radiosity_diff_, np.min(n_integrations), np.max(n_integrations), colors='r', linestyles='--', label='Analytical')
plt.legend()

# %%
rel_error = np.abs((energy_radiosity.T - etc_radiosity_diff_)/etc_radiosity_diff_)
plt.figure()
plt.plot(n_integrations, rel_error*100, 'o-', label=[f'{a}°'for a in angles_list])
plt.legend()

# %%