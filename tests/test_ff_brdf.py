import pyfar as pf
import numpy as np
import pytest
import sparrowpy as sp
import numpy.testing as npt


@pytest.mark.parametrize('function', [
    sp.form_factor.integration.pt_solution,
    sp.form_factor.integration.pt_solution_test,
])
def test_compare_integrations_lambert(function):
    point = np.array([0.5, 0.5, 0.5])
    patch = sp.testing.shoebox_room_stub(1, 1, 1)[0]
    patch_points = patch.pts
    normal = patch.normal

    expected = sp.form_factor.integration.pt_solution(
        point, patch_points)

    actual = function(point, patch_points, normal)

    npt.assert_allclose(actual, expected, rtol=1e-5)


@pytest.mark.parametrize('function', [
    sp.form_factor.integration.pt_solution_test,
])
def test_compare_integrations_lambert_with_brdf(function):
    point = np.array([0.5, 0.5, 0.5])
    patch = sp.testing.shoebox_room_stub(1, 1, 1)[0]
    patch_points = patch.pts
    normal = patch.normal

    expected = sp.form_factor.integration.pt_solution(
        point, patch_points)

    def brdf_lambert(x, y, z):
        return 1


    sampling = pf.samplings.sph_equal_angle(10)
    sampling.weights = pf.samplings.calculate_sph_voronoi_weights(sampling)
    sampling = sampling[sampling.z>0]
    sampling.weights *= 4*np.pi
    brdf_data = sp.brdf.create_from_scattering(sampling, sampling, pf.FrequencyData([0], [100]), brdf_lambert)

    def brdf_specular(x, y, z):
        point = pf.Coordinates(x,y,z)
        index = sampling.find_nearest(point)[0]
        return brdf_data[index]

    actual = function(point, patch_points, normal, brdf_lambert)

    npt.assert_allclose(actual, expected, rtol=1e-5)

