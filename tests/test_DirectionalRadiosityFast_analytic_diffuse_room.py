"""Test radiosity module."""
import numpy as np
import numpy.testing as npt
import pytest
import pyfar as pf

import sparrowpy as sp


@pytest.mark.parametrize('patch_size', [1])
def test_etc_decay(patch_size):
    """Test vs a perfect diffuse room, after Kuttruff."""
    # note that order k=0 means one reflection and k=1 means two reflections
    # (2nd order)
    X = 5
    Y = 6
    Z = 4
    length_histogram = .15
    time_resolution = 1/1000
    max_order_k = 3
    speed_of_sound = 346.18
    receiver = pf.Coordinates(3, 4, 2)
    source = pf.Coordinates(2, 2, 2)

    absorption = 0.1

    sources = pf.Coordinates(0, 0, 1, weights=1)
    receivers = pf.Coordinates(0, 0, 1, weights=1)
    frequencies = np.array([500])
    brdf = sp.brdf.create_from_scattering(
        sources, receivers,
        pf.FrequencyData(1, frequencies),
        pf.FrequencyData(absorption, frequencies))
    walls = sp.testing.shoebox_room_stub(X, Y, Z)

    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        walls, patch_size)

    radiosity.set_wall_brdf(
        np.arange(len(walls)), brdf, sources, receivers)
    radiosity.bake_geometry()
    radiosity.init_source_energy(source)
    radiosity.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        etc_time_resolution=time_resolution,
        etc_duration=length_histogram,
        max_reflection_order=max_order_k,
        recalculate=True)

    etc = radiosity.collect_energy_receiver_mono(receiver)

    # desired
    S = (2*X*Y) + (2*X*Z) + (2*Y*Z)
    A = S*absorption
    alpha_dash = A/S
    V = X*Y*Z
    E_reverb_analytical = 4 / A
    w_0 = E_reverb_analytical/ V # Kuttruff Eq 4.7
    t_0 = 0.03
    t = etc.times
    # Kuttruff Eq 4.10
    reverberation_analytic = w_0 * np.exp(+(
        speed_of_sound*S*np.log(1-alpha_dash)/(4*V))*(t-t_0))

    # compare histogram
    etc_radiosity_db = 10*np.log10(etc.time.flatten())
    etc_analytic_db = 10*np.log10(reverberation_analytic.flatten())

    # compare the diffuse part
    samples_reverb_start = int(0.025/time_resolution)
    samples_reverb_end = int(0.035/time_resolution)
    npt.assert_allclose(
        etc_radiosity_db[samples_reverb_start:samples_reverb_end],
        etc_analytic_db[samples_reverb_start:samples_reverb_end],
        atol=0.3)



@pytest.mark.parametrize('patch_size', [1])
def test_diffuse_energy(patch_size):
    """Test vs a perfect diffuse room, after Kuttruff."""
    # note that order k=0 means one reflection and k=1 means two reflections
    # (2nd order)
    X = 1
    Y = 2
    Z = 3
    length_histogram = 1
    time_resolution = 1
    max_order_k = 3
    speed_of_sound = 343
    receiver = pf.Coordinates(.5, 1, 1.5)
    source = pf.Coordinates(.5, 1, 2)

    absorption = 0.1

    brdf_incoming = pf.Coordinates(0, 0, 1, weights=1)
    brdf_outgoing = pf.Coordinates(0, 0, 1, weights=1)
    frequencies = np.array([500])
    brdf = sp.brdf.create_from_scattering(
        brdf_incoming, brdf_outgoing,
        pf.FrequencyData(1, frequencies),
        pf.FrequencyData(absorption, frequencies))
    walls = sp.testing.shoebox_room_stub(X, Y, Z)

    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        walls, patch_size)

    radiosity.set_wall_brdf(
        np.arange(len(walls)), brdf, brdf_incoming, brdf_outgoing)
    radiosity.bake_geometry()
    radiosity.init_source_energy(source)
    radiosity.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        etc_time_resolution=time_resolution,
        etc_duration=length_histogram,
        max_reflection_order=max_order_k)

    etc = radiosity.collect_energy_receiver_mono(receiver)

    # desired
    S = (2*X*Y) + (2*X*Z) + (2*Y*Z)
    A = S*absorption
    V = X*Y*Z
    E_reverb_analytical = 4 / A
    w_0 = E_reverb_analytical/ V # Kuttruff Eq 4.7

    # compare the diffuse part
    npt.assert_allclose(
        10*np.log10(np.sum(etc.time)),
        10*np.log10(w_0),
        atol=0.3)

