# %%
import numpy as np
import sparrowpy as sp
import pyfar as pf
from datetime import datetime


def run_energy_diff_specular_ratio(width, length, patch_size, source, receiver):
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

    Returns
    -------
    ratio : float
        Ratio of diffuse to specular energy.
    """
    source_is = source.copy()
    source_is.z *= -1
    reflection_len = (receiver - source_is).radius[0]
    speed_of_sound = 343
    sampling_rate = 1
    max_histogram_length = reflection_len / speed_of_sound
    max_histogram_length = 1

    plane = sp.geometry.Polygon(
        np.array(
            [
                [-width / 2, -length / 2, 0],
                [width / 2, -length / 2, 0],
                [width / 2, length / 2, 0],
                [-width / 2, length / 2, 0],
            ],
        ),
        np.array([1, 0, 0]),
        np.array([0, 0, 1]),
    )

    # simulation parameters
    radi = sp.radiosity_fast.DRadiosityFast.from_polygon([plane], patch_size)

    brdf_sources = pf.Coordinates(0, 0, 1, weights=1)
    brdf_receivers = pf.Coordinates(0, 0, 1, weights=1)
    brdf = sp.brdf.create_from_scattering(
        brdf_sources,
        brdf_receivers,
        pf.FrequencyData(1, [100]),
        pf.FrequencyData(0, [100]),
    )

    radi.set_wall_scattering([0], brdf, brdf_sources, brdf_receivers)

    # set air absorption
    radi.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(brdf.frequencies), brdf.frequencies),
    )

    # set absorption coefficient
    radi.set_wall_absorption(
        np.arange(1), pf.FrequencyData(np.zeros_like(brdf.frequencies) + 0, brdf.frequencies),
    )

    # initialize source energy at each patch
    radi.init_source_energy(source.cartesian[0], algorithm="order")

    # gather energy at receiver
    radi.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        histogram_time_resolution=1 / sampling_rate,
        histogram_length=max_histogram_length,
        algorithm="order",
        max_depth=0,
    )
    ir_fast = radi.collect_receiver_energy(
        receiver.cartesian[0],
        speed_of_sound=speed_of_sound,
        histogram_time_resolution=1 / sampling_rate,
        propagation_fx=True,
    )
    I_diffuse = pf.Signal(ir_fast, sampling_rate=sampling_rate)

    I_specular = 1 / (4 * np.pi * reflection_len**2)
    return np.sum(I_diffuse.time) / I_specular

# %%
#run the function
width=100
length=100
patch_size=1
print(f"Width: {width}\nLength: {length}\nPatch size: {patch_size}")
start = datetime.now()
cal_ratio = run_energy_diff_specular_ratio(
    width,
    length,
    patch_size,
    source=pf.Coordinates(0, 0, 1),
    receiver=pf.Coordinates(0, 0, 1),
)
delta = datetime.now() - start
print(f"Time elapsed: {delta}\nValue: {cal_ratio}")

"""
Width: 100
Length: 100
Patch size: 5
Time elapsed: 0:00:00.506815
Value: 0.34191990075547674

Width: 100
Length: 100
Patch size: 3
Time elapsed: 0:00:02.581830
Value: 1.403211682942388

Width: 100
Length: 100
Patch size: 2
Time elapsed: 0:00:11.878616
Value: 1.1414845658385908

Width: 100
Length: 100
Patch size: 1
Time elapsed: 0:02:57.446832
Value: 1.749127668862095
"""
