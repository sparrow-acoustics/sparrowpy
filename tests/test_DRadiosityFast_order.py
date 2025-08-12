"""Test radiosity module."""
import numpy as np
import numpy.testing as npt
import pytest
import pyfar as pf

import sparrowpy as sp



@pytest.mark.parametrize('walls', [
    # perpendicular walls
    [0, 2],
    # parallel walls
    [0, 1],
    ])
@pytest.mark.parametrize('patch_size', [
    0.5,
    ])
def test_form_factors_directivity_for_diffuse(
        sample_walls, walls, patch_size, sofa_data_diffuse):
    """Test form factor calculation for perpendicular walls."""
    wall_source = sample_walls[walls[0]]
    wall_receiver = sample_walls[walls[1]]
    walls = [wall_source, wall_receiver]

    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        walls, patch_size)
    data, sources, receivers = sofa_data_diffuse
    radiosity.set_wall_brdf(
        np.arange(len(walls)), data, sources, receivers)
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.bake_geometry()
    npt.assert_almost_equal(radiosity._form_factors_tilde.shape, (8, 8, 4, 4))
    # test _form_factors_tilde
    for i in range(radiosity._form_factors_tilde.shape[0]):
        for j in range(radiosity._form_factors_tilde.shape[0]):
            if i < j:
                npt.assert_almost_equal(
                    radiosity._form_factors_tilde[i, j, :, :],
                    radiosity._form_factors[i, j])
            else:
                npt.assert_almost_equal(
                    radiosity._form_factors_tilde[i, j, :, :],
                    radiosity._form_factors[j, i])


@pytest.mark.parametrize('patch_size', [1])
def test_order_vs_analytic(patch_size):
    """Test vs a perfect diffuse room, after Kuttruff."""
    # note that order k=0 means one reflection and k=1 means two reflections
    # (2nd order)
    X = 5
    Y = 3
    Z = 4
    length_histogram = .15
    time_resolution = 1/500
    max_order_k = 3
    speed_of_sound = 346.18
    receiver = pf.Coordinates(3, 1.5, 2)
    source = pf.Coordinates(2, 1.5, 2)

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
    radiosity.set_air_attenuation(
        pf.FrequencyData(
            np.zeros_like(brdf.frequencies),
            brdf.frequencies))
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
    npt.assert_allclose(
        etc_radiosity_db[5:19],
        etc_analytic_db[5:19],
        atol=2) # maximal 2dB difference
