"""
This files validates the Radiosity method for a diffuse infinite plane.

The analytical results are taken from Svensson et. al. [1].

[1] U. P. Svensson and L. Savioja, "The Lambert diffuse reflection model
revisited," The Journal of the Acoustical Society of America, vol. 156,
no. 6, pp. 3788-3796, Dec. 2024, doi: 10.1121/10.0034561.

"""

import numpy as np
import sparrowpy as sp
import pyfar as pf
import pytest
import numpy.testing as npt


def run_energy_diff_specular_ratio(
        width, length, patch_size, source, receiver):
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
    reflection_len =  (receiver - source_is).radius[0]
    speed_of_sound = 343
    sampling_rate = 1
    etc_duration = reflection_len/speed_of_sound
    etc_duration=1

    plane = sp.geometry.Polygon(
            [[-width/2, -length/2, 0],
             [width/2, -length/2, 0],
             [width/2, length/2, 0],
             [-width/2, length/2, 0]],
            [1, 0, 0], [0, 0, 1])

    #simulation parameters
    radi = sp.DirectionalRadiosityFast.from_polygon(
        [plane], patch_size)

    brdf_sources = pf.Coordinates(0, 0, 1, weights=1)
    brdf_receivers = pf.Coordinates(0, 0, 1, weights=1)
    brdf = sp.brdf.create_from_scattering(
        brdf_sources,
        brdf_receivers,
        pf.FrequencyData(1, [100]),
        pf.FrequencyData(0, [100]),
    )

    radi.set_wall_brdf(
        np.arange(1), brdf, brdf_sources, brdf_receivers)

    # set air absorption
    radi.set_air_attenuation(
        pf.FrequencyData(
            np.zeros_like(brdf.frequencies),
            brdf.frequencies))

    # initialize source energy at each patch
    radi.init_source_energy(source)

    # gather energy at receiver
    radi.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        etc_time_resolution=1/sampling_rate,
        etc_duration=etc_duration,
        max_reflection_order=0)

    I_diffuse = radi.collect_energy_receiver_mono(receiver)

    I_specular = 1/(4*np.pi*reflection_len**2)
    return np.sum(I_diffuse.time)/I_specular


@pytest.mark.parametrize("patch_size", [1])
def test_colocated_source_receiver(patch_size):
    """
    Compares test case 1 from [1], where source and receiver are located
    at the same position. For this case the Energy ratio diffuse to
    specular should be 2.
    """
    width = 20
    depth = 20
    source = pf.Coordinates(0, 0, 3, weights=1)
    receiver = pf.Coordinates(0, 0, 3, weights=1)
    ratio = run_energy_diff_specular_ratio(
        width, depth, patch_size, source, receiver)

    # the energy should be 2 ideally, but the simulation cannot cover
    # an infinite plane, therefore the energy is slightly less than 2
    npt.assert_array_less(ratio, 2)
    npt.assert_allclose(ratio, 1.97, rtol=0.01)


@pytest.mark.parametrize("patch_size", [1])
def test_source_receiver_along_same_normal(patch_size):
    """
    Compares test case 2 from [1], where Source and receiver are along
    the same normal. For this case the Energy ratio diffuse to
    specular should be 2.
    """
    width = 20
    depth = 20
    source = pf.Coordinates(0, 0, 5, weights=1)
    receiver = pf.Coordinates(0, 0, 3, weights=1)
    ratio = run_energy_diff_specular_ratio(
        width, depth, patch_size, source, receiver)

    # the energy should be 2 ideally, but the simulation cannot cover
    # an infinite plane, therefore the energy is slightly less than 2
    npt.assert_array_less(ratio, 2)
    npt.assert_allclose(ratio, 1.97, rtol=0.01)


@pytest.mark.parametrize("theta_deg", [30, 45, 60])
@pytest.mark.parametrize("patch_size", [1])
def test_source_receiver_same_hight(patch_size, theta_deg):
    """
    Compares test case 3 from [1], where receiver is at the same height as
    the source, but moved sideways/laterally. For this case the Energy
    ratio diffuse to specular should be 2cos(theta). With theta being the
    incident angle of the specular reflection.
    """
    width = 20
    depth = 20
    theta_rad = np.deg2rad(theta_deg)
    source = pf.Coordinates.from_spherical_colatitude(
        0, theta_rad, 2, weights=1)
    receiver = pf.Coordinates.from_spherical_colatitude(
        np.pi, theta_rad, 2, weights=1)
    ratio = run_energy_diff_specular_ratio(
        width, depth, patch_size, source, receiver)

    npt.assert_allclose(ratio, 2*np.cos(theta_rad), atol=0.03, rtol=0.03)

