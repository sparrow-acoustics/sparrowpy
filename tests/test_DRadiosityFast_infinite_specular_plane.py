"""
This files validates the Radiosity method for a diffuse infinite plane.

The analytical results are taken from Svensson et. al. [1].

[1] U. P. Svensson and L. Savioja, "The Lambert diffuse reflection model
revisited," The Journal of the Acoustical Society of America, vol. 156,
no. 6, pp. 3788–3796, Dec. 2024, doi: 10.1121/10.0034561.

"""

import numpy as np
import sparrowpy as sp
import pyfar as pf
import pytest
import numpy.testing as npt


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

    I_diffuse = 2/(4*np.pi*reflection_len**2)
    return I_diffuse/np.sum(I_specular.freq)


@pytest.mark.parametrize("brdf_coords", [
    pf.Coordinates(0, 0, 1, weights=1),
    pf.samplings.sph_equal_angle(30),
    pf.samplings.sph_equal_angle(15),
])
@pytest.mark.parametrize("patch_size", [
    1, 3,
])
def test_colocated_source_receiver(brdf_coords, patch_size):
    """
    Compares test case 1 from [1], where source and receiver are located
    at the same position. For this case the Energy ratio diffuse to
    specular should be 2.
    """
    width = 3
    depth = 3
    if brdf_coords.csize > 1:
        brdf_coords.weights = pf.samplings.calculate_sph_voronoi_weights(
            brdf_coords)
        brdf_coords = brdf_coords[brdf_coords.z>0]
    source = pf.Coordinates(0, 0, 3, weights=1)
    receiver = pf.Coordinates(0, 0, 3, weights=1)
    ratio = calculate_ratio_new(
        width, depth, patch_size, source, receiver, brdf_coords)

    # the energy should be 2 ideally, but the simulation cannot cover
    # an infinite plane, therefore the energy is slightly less than 2
    npt.assert_array_less(ratio, 2)
    npt.assert_allclose(ratio, 2, rtol=0.01)



@pytest.mark.parametrize("brdf_coords", [
    pf.Coordinates(0, 0, 1, weights=1),
    pf.samplings.sph_equal_angle(30),
])
@pytest.mark.parametrize("patch_size", [
    1, .5,
])
def test_source_receiver_along_same_normal(brdf_coords, patch_size):
    """
    Compares test case 2 from [1], where Source and receiver are along
    the same normal. For this case the Energy ratio diffuse to
    specular should be 2.
    """
    width = 1
    depth = 1
    if brdf_coords.csize > 1:
        brdf_coords.weights = pf.samplings.calculate_sph_voronoi_weights(
            brdf_coords)
        brdf_coords = brdf_coords[brdf_coords.z>0]
    source = pf.Coordinates(0, 0, 5, weights=1)
    receiver = pf.Coordinates(0, 0, 3, weights=1)
    ratio = calculate_ratio_new(
        width, depth, patch_size, source, receiver, brdf_coords)

    # the energy should be 2 ideally, but the simulation cannot cover
    # an infinite plane, therefore the energy is slightly less than 2
    npt.assert_array_less(ratio, 2)
    npt.assert_allclose(ratio, 2, rtol=0.01)


@pytest.mark.parametrize("theta_deg", [30, 45, 60])
@pytest.mark.parametrize("brdf_coords", [
    pf.Coordinates(0, 0, 1, weights=1),
    pf.samplings.sph_equal_angle(30),
])
@pytest.mark.parametrize("patch_size", [
    1, .5,
])
def test_source_receiver_same_hight(patch_size, theta_deg, brdf_coords):
    """
    Compares test case 3 from [1], where receiver is at the same height as
    the source, but moved sideways/laterally. For this case the Energy
    ratio diffuse to specular should be 2cos(theta). With theta being the
    incident angle of the specular reflection.
    """
    width = 1
    depth = 1
    if brdf_coords.csize > 1:
        brdf_coords.weights = pf.samplings.calculate_sph_voronoi_weights(
            brdf_coords)
        brdf_coords = brdf_coords[brdf_coords.z>0]
    theta_rad = np.deg2rad(theta_deg)
    source = pf.Coordinates.from_spherical_colatitude(
        0, theta_rad, 2, weights=1)
    receiver = pf.Coordinates.from_spherical_colatitude(
        np.pi, theta_rad, 2, weights=1)
    ratio = calculate_ratio_new(
        width, depth, patch_size, source, receiver, brdf_coords)

    npt.assert_allclose(ratio, 2*np.cos(theta_rad), atol=0.03, rtol=0.03)

