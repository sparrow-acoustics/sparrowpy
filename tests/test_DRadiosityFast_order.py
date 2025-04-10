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
