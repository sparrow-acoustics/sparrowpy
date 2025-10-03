import pyfar as pf
import numpy as np
import pytest
import sparrowpy as sp
import numpy.testing as npt
import sparrowpy.form_factor


INTEGRATION_FUNCS = [
    sp.form_factor.integration.pt_solution,
    sp.form_factor.integration.point_patch_factor_dblquad,
    sp.form_factor.integration.point_patch_factor_leggaus_planar,
    sp.form_factor.integration.point_patch_factor_mc_planar,
]
@pytest.mark.parametrize('function', INTEGRATION_FUNCS)
def compare_integrations_lambert(function,patch_option):
    point = np.array([0.5, 0.5, 0.5])
    patch = sp.testing.shoebox_room_stub(3, 3, 3)[patch_option]
    patch_points = patch.pts
    normal = patch.normal

    expected = sp.form_factor.integration.pt_solution(
        point, patch_points)
    
    ##due to different input args, need to handle separately
    if function == sp.form_factor.integration.pt_solution:
        actual = function(point, patch_points)
    else:
        actual = function(point, patch_points, normal)
    npt.assert_allclose(actual, expected, rtol=1e-1)

@pytest.mark.parametrize('function', INTEGRATION_FUNCS)
def test_compare_integrations_lambert_0(function):
    compare_integrations_lambert(function,0)

@pytest.mark.parametrize('function', INTEGRATION_FUNCS)
def test_compare_integrations_lambert_1(function):
    compare_integrations_lambert(function,1)
    
@pytest.mark.parametrize('function', INTEGRATION_FUNCS)
def test_compare_integrations_lambert_2(function):
    compare_integrations_lambert(function,2)

def test_compare_integrations_with_brdf():
    point = pf.Coordinates.from_cartesian(0.5,0.5,0.5)
    patch = sp.testing.shoebox_room_stub(1, 1, 1)[2]
    patch_points = patch.pts
    normal = patch.normal
    patch_size = 0.5

    expected = sp.form_factor.integration.pt_solution(
        point, patch_points)

    sampling = pf.samplings.sph_equal_angle(10)
    sampling.weights = pf.samplings.calculate_sph_voronoi_weights(sampling)
    sampling = sampling[sampling.z>0]
    sampling.weights *= 4*np.pi
    scattering_coef = 1
    brdf_data = sp.brdf.create_from_scattering(sampling, sampling, pf.FrequencyData([scattering_coef,0], [100,500]))

    radi = sp.DirectionalRadiosityFast.from_polygon([patch], patch_size)
    radi.set_wall_brdf(
        np.arange(1), brdf_data, sampling, sampling)

    # set air absorption
    radi.set_air_attenuation(
        pf.FrequencyData(
            np.zeros_like(brdf_data.frequencies),
            brdf_data.frequencies))

    radi.init_source_energy(point)
    energy_0_dir = radi._energy_init_source

    #########Integration with BRDF#########
   
    radi_BRDF = sp.DirectionalRadiosityFast.from_polygon([patch], patch_size)
    radi_BRDF.set_wall_brdf(
        np.arange(1), brdf_data, sampling, sampling)

    # set air absorption
    radi_BRDF.set_air_attenuation(
        pf.FrequencyData(
            np.zeros_like(brdf_data.frequencies),
            brdf_data.frequencies))
    radi_BRDF.init_source_energy_withBRDFIntegration(point)
    energy_0_dir_BRDF = radi_BRDF._energy_init_source

    #assert the BRDF integration gives same result as lambertian    
    npt.assert_allclose(energy_0_dir_BRDF[:,:,0], energy_0_dir[:,:,0], rtol=1e-1)
    #assert the BRDF integration max index is same as the specular direction
    idx_max_dir = np.argmax(energy_0_dir[0:4, :, 1], axis=1)
    idx_max_dir_BRDF = np.argmax(energy_0_dir_BRDF[0:4, :, 1], axis=1)
    npt.assert_array_equal(idx_max_dir, idx_max_dir_BRDF)

    #def brdf_specular(x, y, z):
    #    point = pf.Coordinates(x,y,z)
    #    index = sampling.find_nearest(point)[0]
    #    return brdf_data[index]

    #actual = function(point, patch_points, normal, brdf_lambert)

    #npt.assert_allclose(actual, expected, rtol=1e-5)
#    
#
## Example functions to test
#def functionA(x):
#    return x + 2
#
#def functionB(x):
#    return x * 3
#
#@pytest.mark.parametrize('function', [functionA, functionB])
#def test_compare_methods(function, input_value):
#    result = function(input_value)
#    # Example assertion: just check result is not None
#    assert result is not None
#
#def test_compare_integrations_XXXXXX(patch_energy_normal_direction_0_degree):
#    point = np.array([0.5, 0.5, 0.5])
#    #vector quadrature to source point
#    patch = sp.testing.shoebox_room_stub(1, 1, 1)[0]
#    patch_points = patch.pts
#    normal = patch.normal
#    brdf_data = patch_energy_normal_direction_0_degree(point[0], point[1], point[2])
#
#    check_data = brdf_data

def test_energy_conservation():
    sampling = sp.brdf.sph_gaussian_hemisphere(sh_order=20)
    scattering_coef = 0.0
    brdf_data = sp.brdf.create_from_scattering(sampling, sampling, pf.FrequencyData([scattering_coef], [100]))
    sum = 0
    for i in range(sampling.cartesian.shape[0]):
        cos_theta = np.dot([0,0,1],sampling.cartesian[i])
        sum += brdf_data.freq[i,0] * cos_theta * sampling.weights[i]
    Energy_conservation = sum
    print(f'Energy Conserv = {Energy_conservation}, should be ~= {1}')