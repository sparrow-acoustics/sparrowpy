# %%
"""
This files validates the Radiosity method for a diffuse infinite plane.

The analytical results are taken from Svensson et. al. [1].

[1] U. P. Svensson and L. Savioja, "The Lambert diffuse reflection model
revisited," The Journal of the Acoustical Society of America, vol. 156,
no. 6, pp. 3788–3796, Dec. 2024, doi: 10.1121/10.0034561.

"""

import matplotlib.pyplot as plt
import numpy as np
import sparrowpy as sp
import pyfar as pf
import pytest
import numpy.testing as npt

# %%
def calculate_ratio_new(
        width, length, patch_size, source, receiver, brdf_coords):
    """
    Calculate the ratio of diffuse to specular energy for an plane.
    The plane is located in the x-y plane. Its center is at (0, 0, 0).

    Parameters
    ----------
    width : float
        Width of the plane.
    length : float
        length of the plane.
    patch_size : float
        Size of the patches.
    source : pf.Coordinates
        Position of the source.
    receiver : pf.Coordinates
        Position of the receiver in cartesian.
    brdf_coords : pf.Coordinates
        Coordinates for the BRDF construction.

    Returns
    -------
    ratio : float
        Ratio of diffuse to specular energy.
    """
    source_is = source.copy()
    source_is.z *= -1
    reflection_len =  (receiver - source_is).radius[0]
    speed_of_sound = 343
    sampling_rate = 1
    max_histogram_length = reflection_len/speed_of_sound
    max_histogram_length=1


    plane = sp.geometry.Polygon(
            [[-width/2, -length/2, 0],
             [width/2, -length/2, 0],
             [width/2, length/2, 0],
             [-width/2, length/2, 0]],
            [1, 0, 0], [0, 0, 1])

    #simulation parameters
    radi = sp.radiosity_fast.DRadiosityFast.from_polygon(
        [plane], patch_size)

    brdf_sources = brdf_coords.copy()
    brdf_sources.weights = np.sin(brdf_sources.elevation)
    brdf_receivers = brdf_sources.copy()
    brdf = sp.brdf.create_from_scattering(
        brdf_sources,
        brdf_receivers,
        pf.FrequencyData(0, [100]),
        pf.FrequencyData(0, [100]),
    )

    radi.set_wall_scattering(
        np.arange(1), brdf, brdf_sources, brdf_receivers)

    # set air absorption
    radi.set_air_attenuation(
        pf.FrequencyData(
            np.zeros_like(brdf.frequencies),
            brdf.frequencies))

    # set absorption coefficient
    radi.set_wall_absorption(
        np.arange(1),
        pf.FrequencyData(
            np.zeros_like(brdf.frequencies)+0,
            brdf.frequencies))

    # initialize source energy at each patch
    radi.init_source_energy(source.cartesian[0], algorithm='order')

    # gather energy at receiver
    radi.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        histogram_time_resolution=1/sampling_rate,
        histogram_length=max_histogram_length,
        algorithm='order', max_depth=0)
    ir_fast = radi.collect_receiver_energy(
        receiver.cartesian[0], speed_of_sound=speed_of_sound,
        histogram_time_resolution=1/sampling_rate,
        propagation_fx=True)
    I_specular = pf.Signal(ir_fast, sampling_rate=sampling_rate)

    I_diffuse = 1/(2*np.pi*(reflection_len)**2)
    return I_diffuse/np.sum(I_specular.freq)

delta_angles = [60, 45, 30, 10, 5]

brdf_coordss = [
        pf.samplings.sph_equal_angle(ang)
        for ang in delta_angles
        ]

patch_sizes = [1, 1/3, 1/5]
weights = []
radios = np.zeros((len(brdf_coordss), len(patch_sizes)))
for i_brdf, brdf_coords in enumerate(brdf_coordss):
    if brdf_coords.weights is None:
        brdf_coords.weights = pf.samplings.calculate_sph_voronoi_weights(
            brdf_coords)
        brdf_coords = brdf_coords[brdf_coords.z>0]
    weights.append(brdf_coords.weights[0])
    for i_patch, patch_size in enumerate(patch_sizes):
        width = 10
        depth = 10
        source = pf.Coordinates(0, 0, 3, weights=1)
        receiver = pf.Coordinates(0, 0, 3, weights=1)
        ratio = calculate_ratio_new(
            width, depth, patch_size, source, receiver, brdf_coords)
        radios[i_brdf, i_patch] = ratio
weights = np.array(weights)[:, np.newaxis]
radios
# %%
plt.figure()
plt.imshow(radios, vmin=0, vmax=4, cmap='coolwarm')
plt.colorbar()
radios
# %%
