# %%
import sparrowpy as sp
import numpy as np
import pyfar as pf
import sofar as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

# %%
# Define parameters
patch_size = 1
etc_duration = 1
etc_time_resolution = 1/2
max_reflection_order = 1
speed_of_sound = 343.2
absorption = 0

width = patch_size
length = patch_size

# Task configuration
angles_list = [30, 22.5,15,10]
n_integrations = np.array([100, 250, 500, 750, 1000])
n_integrations = np.sqrt(n_integrations).astype(int)

def _compute_energy_for_task(args):
    """
    Worker to compute energy for a given (angles, n_integration).
    Constructs geometry and BRDF per process to avoid shared mutable state.
    Returns (angles, n_integration, energy_sum).
    """
    angles, n_integration, patch_size_local = args

    # geometry
    plane = sp.geometry.Polygon(
        [[-patch_size_local/2, -patch_size_local/2, 0],
         [ patch_size_local/2, -patch_size_local/2, 0],
         [ patch_size_local/2,  patch_size_local/2, 0],
         [-patch_size_local/2,  patch_size_local/2, 0]],
        [1, 0, 0], [0, 0, 1]
    )

    source = pf.Coordinates(1, 0, 1)
    receiver = pf.Coordinates(-1, 0, 1)

    # radiosity object
    radiosity_fast = sp.DirectionalRadiosityFast.from_polygon([plane], patch_size_local)

    # BRDF samplings
    brdf_sources = pf.samplings.sph_equal_angle(angles)
    brdf_sources.weights = pf.samplings.calculate_sph_voronoi_weights(brdf_sources)
    brdf_sources = brdf_sources[brdf_sources.z > 0]

    brdf_receivers = pf.samplings.sph_equal_angle(angles)
    brdf_receivers.weights = pf.samplings.calculate_sph_voronoi_weights(brdf_receivers)
    brdf_receivers = brdf_receivers[brdf_receivers.z > 0]

    frequencies = np.array([1000])
    brdf = sp.brdf.create_from_scattering(
        brdf_sources,
        brdf_receivers,
        pf.FrequencyData(0, frequencies),
        pf.FrequencyData(absorption, frequencies),
    )

    # set BRDF and air attenuation
    radiosity_fast.set_wall_brdf(np.arange(1), brdf, brdf_sources, brdf_receivers)
    radiosity_fast.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(brdf.frequencies), brdf.frequencies)
    )

    # Monte Carlo settings
    radiosity_fast._integration_method = "leggauss"
    radiosity_fast._integration_sampling = n_integration

    # compute energy at receiver
    radiosity_fast.init_source_energy_brdf_integration(source)
    radiosity_fast.calculate_energy_exchange(
        speed_of_sound=343.2,
        etc_time_resolution=1/2,
        etc_duration=1,
        max_reflection_order=0,
    )
    etc_radiosity = radiosity_fast.collect_energy_receiver_mono_brdf_integration(receivers=receiver)

    energy = np.sum(etc_radiosity.time)
    return angles, n_integration, energy

if __name__ == "__main__":
    # geometry for analytical reference
    source = pf.Coordinates(1, 0, 1)
    receiver = pf.Coordinates(-1, 0, 1)

    # dispatch all (angles, n_integration) tasks
    tasks = [(ang, nint, patch_size) for ang, nint in product(angles_list, n_integrations)]

    results = {}
    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(_compute_energy_for_task, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            angles, n_integration, energy = fut.result()
            results[(angles, n_integration)] = energy

    # assemble results into arrays of shape (len(angles_list), len(n_integrations))
    energy_radiosity = np.zeros((len(angles_list), len(n_integrations)))
    for i, angles in enumerate(angles_list):
        for j, nint in enumerate(n_integrations):
            energy_radiosity[i, j] = results[(angles, nint)]

    energy_analytical = 1 / (4 * np.pi * ((receiver + source).radius[0])**2)

    # Plot results
    plt.figure()
    for i, angles in enumerate(angles_list):
        plt.plot(n_integrations, energy_radiosity[i], 'o-', label=f'{angles}°')
    plt.hlines(energy_analytical, np.min(n_integrations), np.max(n_integrations),
               colors='r', linestyles='--', label='Analytical')
    plt.legend()

    rel_error = np.abs((energy_radiosity - energy_analytical) / energy_analytical)
    plt.figure()
    for i, angles in enumerate(angles_list):
        plt.plot(n_integrations, rel_error[i] * 100, 'o-', label=f'{angles}°')
    plt.legend()
