import pyfar as pf
import numpy as np
import pytest
import sparrowpy as sp
import numpy.testing as npt
import sparrowpy.form_factor


@pytest.mark.parametrize('quadrature', [
   1, 4, 8, 16
    ])
@pytest.mark.parametrize('patch_size', [
    2, 1
    ])
@pytest.mark.parametrize('sh_order', [
    6, 16, 24, 30
    ])
@pytest.mark.parametrize('elevation', [
    15, 30, 45, 60, 75, 89
    ])
@pytest.mark.parametrize('src_rcv_radiusfromcenter', [
    6
    ])
def test_BRDF_diffuse_leggauss_vs_original(quadrature, patch_size, sh_order, elevation, src_rcv_radiusfromcenter):
    distance_from_patch = src_rcv_radiusfromcenter

    source = pf.Coordinates.from_spherical_elevation(0,np.deg2rad(elevation),distance_from_patch)
    receiver = pf.Coordinates.from_spherical_elevation(np.pi,source.elevation[0],source.radius[0])
    total_distance = source.radius[0] + receiver.radius[0]
    analytical = 1/(4*np.pi*total_distance**2)
    
    width = 2
    length = 2
    patch = sp.geometry.Polygon(
            [[-width/2, -length/2, 0],
                [width/2, -length/2, 0],
                [width/2, length/2, 0],
                [-width/2, length/2, 0]],
            [1, 0, 0], [0, 0, 1])

    sampling_rate = 4000/2
    speed_of_sound = 343.2
    etc_duration = 0.5
    etc_time = etc_duration
    toplot_receiver = receiver
    sampling = sp.brdf.sph_gaussian_hemisphere(sh_order=sh_order)
    scattering_coef = 1 #diffuse

    brdf_data = sp.brdf.create_from_scattering(sampling, sampling, pf.FrequencyData([scattering_coef], [300]))

    #original method
    radi = sp.DirectionalRadiosityFast.from_polygon([patch], patch_size)
    radi.set_wall_brdf(
        np.arange(1), brdf_data, sampling, sampling)

    # set air absorption
    radi.set_air_attenuation(
        pf.FrequencyData(
            np.zeros_like(brdf_data.frequencies),
            brdf_data.frequencies))
    radi.init_source_energy(source)


    radi.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        etc_time_resolution=1/sampling_rate,
        etc_duration=etc_duration,
        max_reflection_order=0)

    etc = radi.collect_energy_receiver_mono(toplot_receiver, False)
    max_val = np.max(etc.time)
    #########Integration with BRDF#########

    radi_brdf_integration = sp.DirectionalRadiosityFast.from_polygon([patch], patch_size)
    radi_brdf_integration.set_wall_brdf(
        np.arange(1), brdf_data, sampling, sampling)

    radi_brdf_integration._integration_method = "leggauss"
    radi_brdf_integration._integration_sampling = quadrature
    # set air absorption
    radi_brdf_integration.set_air_attenuation(
        pf.FrequencyData(
            np.zeros_like(brdf_data.frequencies),
            brdf_data.frequencies))
    radi_brdf_integration.init_source_energy_brdf_integration(source)



    radi_brdf_integration.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        etc_time_resolution=1/sampling_rate,
        etc_duration=etc_duration,
        max_reflection_order=0)

    etc_brdf_integration = radi_brdf_integration.collect_energy_receiver_mono_brdf_integration(toplot_receiver, False)

    max_val_brdf_integration = np.max(etc_brdf_integration.time)
    

    print(f'max_val = {max_val}, should be ~= {max_val_brdf_integration}')
    npt.assert_allclose(max_val, max_val_brdf_integration, rtol=0.1)


@pytest.mark.parametrize('elevation', [
    15, 30, 45, 60, 75, 89
    ])
@pytest.mark.parametrize('quadrature', [
   1, 4, 8, 16
    ])
@pytest.mark.parametrize('sh_order', [
    6, 16, 24, 30
    ])
def test_BRDF_specular_leggauss_vs_analytical_6m_radius_PS_2m(quadrature, sh_order, elevation):
    distance_from_patch = 6

    source = pf.Coordinates.from_spherical_elevation(0,np.deg2rad(elevation),distance_from_patch)
    receiver = pf.Coordinates.from_spherical_elevation(np.pi,source.elevation[0],source.radius[0])
    total_distance = source.radius[0] + receiver.radius[0]
    analytical = 1/(4*np.pi*total_distance**2)
    
    width = 2
    length = 2
    patch = sp.geometry.Polygon(
            [[-width/2, -length/2, 0],
                [width/2, -length/2, 0],
                [width/2, length/2, 0],
                [-width/2, length/2, 0]],
            [1, 0, 0], [0, 0, 1])

    sampling_rate = 4000/2
    speed_of_sound = 343.2
    etc_duration = 0.5
    etc_time = etc_duration
    toplot_receiver = receiver
    sampling = sp.brdf.sph_gaussian_hemisphere(sh_order=sh_order)
    scattering_coef = 0 #specular
    patch_size = 2

    brdf_data = sp.brdf.create_from_scattering(sampling, sampling, pf.FrequencyData([scattering_coef], [300]))

    #########Integration with BRDF#########

    radi_brdf_integration = sp.DirectionalRadiosityFast.from_polygon([patch], patch_size)
    radi_brdf_integration.set_wall_brdf(
        np.arange(1), brdf_data, sampling, sampling)

    radi_brdf_integration._integration_method = "leggauss"
    radi_brdf_integration._integration_sampling = quadrature
    # set air absorption
    radi_brdf_integration.set_air_attenuation(
        pf.FrequencyData(
            np.zeros_like(brdf_data.frequencies),
            brdf_data.frequencies))
    radi_brdf_integration.init_source_energy_brdf_integration(source)

    radi_brdf_integration.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        etc_time_resolution=1/sampling_rate,
        etc_duration=etc_duration,
        max_reflection_order=0)

    etc_brdf_integration = radi_brdf_integration.collect_energy_receiver_mono_brdf_integration(toplot_receiver, False)

    max_val_brdf_integration = np.max(etc_brdf_integration.time)
    

    print(f'max_val = {max_val_brdf_integration}, should be ~= {analytical}')
    npt.assert_allclose(max_val_brdf_integration, analytical, rtol=0.10) #10% tolerance


@pytest.mark.parametrize('elevation', [
    15, 30, 45, 60, 75, 89
    ])
@pytest.mark.parametrize('quadrature', [
   8
    ])
@pytest.mark.parametrize('sh_order', [
    24
    ])
def test_BRDF_specular_leggauss_vs_analytical_6m_radius_ps2m_q8_sh_24(quadrature, sh_order, elevation):
    distance_from_patch = 6

    source = pf.Coordinates.from_spherical_elevation(0,np.deg2rad(elevation),distance_from_patch)
    receiver = pf.Coordinates.from_spherical_elevation(np.pi,source.elevation[0],source.radius[0])
    total_distance = source.radius[0] + receiver.radius[0]
    analytical = 1/(4*np.pi*total_distance**2)
    
    width = 2
    length = 2
    patch = sp.geometry.Polygon(
            [[-width/2, -length/2, 0],
                [width/2, -length/2, 0],
                [width/2, length/2, 0],
                [-width/2, length/2, 0]],
            [1, 0, 0], [0, 0, 1])

    sampling_rate = 4000/2
    speed_of_sound = 343.2
    etc_duration = 0.5
    etc_time = etc_duration
    toplot_receiver = receiver
    sampling = sp.brdf.sph_gaussian_hemisphere(sh_order=sh_order)
    scattering_coef = 0 #specular
    patch_size = 2

    brdf_data = sp.brdf.create_from_scattering(sampling, sampling, pf.FrequencyData([scattering_coef], [300]))

    #########Integration with BRDF#########

    radi_brdf_integration = sp.DirectionalRadiosityFast.from_polygon([patch], patch_size)
    radi_brdf_integration.set_wall_brdf(
        np.arange(1), brdf_data, sampling, sampling)

    radi_brdf_integration._integration_method = "leggauss"
    radi_brdf_integration._integration_sampling = quadrature
    # set air absorption
    radi_brdf_integration.set_air_attenuation(
        pf.FrequencyData(
            np.zeros_like(brdf_data.frequencies),
            brdf_data.frequencies))
    radi_brdf_integration.init_source_energy_brdf_integration(source)

    radi_brdf_integration.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        etc_time_resolution=1/sampling_rate,
        etc_duration=etc_duration,
        max_reflection_order=0)

    etc_brdf_integration = radi_brdf_integration.collect_energy_receiver_mono_brdf_integration(toplot_receiver, False)

    max_val_brdf_integration = np.max(etc_brdf_integration.time)
    

    print(f'max_val = {max_val_brdf_integration}, should be ~= {analytical}')
    npt.assert_allclose(max_val_brdf_integration, analytical, rtol=0.3) #10% tolerance