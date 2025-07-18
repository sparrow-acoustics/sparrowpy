
"""runs simulation"""
import numpy as np
import sparrowpy as sp
import pyfar as pf
import tracemalloc
import pyrato

def run_simu_mem(walls, source, receiver,
             patch_size=1, absorption=.1, scattering=1,
             speed_of_sound=343.26, time_step=.1, duration=.5,
             refl_order=3, freq=np.array([1000]), res=30):

    att=pyrato.air_attenuation_coefficient(freq)
    att_np= att* 0.115129254650564

    # set brdfs
    samples = pf.samplings.sph_equal_angle(delta_angles=res)
    samples.weights=np.ones(samples.cshape[0])

    brdf_sources = samples[np.where((samples.elevation*180/np.pi >= 0))].copy()
    brdf_receivers = samples[np.where((samples.elevation*180/np.pi >= 0))].copy()

    # create object
    tracemalloc.start()
    radi = sp.DirectionalRadiosityFast.from_polygon(walls, patch_size)

    brdf = sp.brdf.create_from_scattering(
        brdf_sources,
        brdf_receivers,
        pf.FrequencyData(scattering, freq),
        pf.FrequencyData(absorption, freq))

    # set directional scattering data
    radi.set_wall_brdf(
        np.arange(len(walls)), brdf, brdf_sources, brdf_receivers)
    # set air absorption
    radi.set_air_attenuation(
        pf.FrequencyData(
            att_np,
            brdf.frequencies))

    mem_scene = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()

    # calculate from factors including brdfs
    radi.bake_geometry()

    mem_bake =  tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()

    radi.init_source_energy(source)

    radi.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        etc_time_resolution=time_step,
        etc_duration=duration,
        max_reflection_order=refl_order)

    mem_exchange = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()

    etc_radiosity = radi.collect_energy_receiver_mono(receivers=receiver)


    mem_collect = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return [mem_scene, mem_bake, mem_exchange, mem_collect]

def run_simu(walls, source, receiver,
             patch_size=1, absorption=.1, scattering=1,
             speed_of_sound=343.26, time_step=.1, duration=.5,
             refl_order=3, freq=np.array([1000]), res=30):

    att=10*np.log10(pyrato.air_attenuation_coefficient(freq))/1000
    att_np= att* 0.115129254650564

    # set brdfs
    samples = pf.samplings.sph_equal_angle(delta_angles=res)
    samples.weights=np.ones(samples.cshape[0])

    brdf_sources = samples[np.where((samples.elevation*180/np.pi >= 0))].copy()
    brdf_receivers = samples[np.where((samples.elevation*180/np.pi >= 0))].copy()

    # create object
    radi = sp.DirectionalRadiosityFast.from_polygon(walls, patch_size)

    # create directional scattering data (totally diffuse)
    brdf = sp.brdf.create_from_scattering(
        brdf_sources,
        brdf_receivers,
        pf.FrequencyData(scattering, freq),
        pf.FrequencyData(absorption, freq))

    # set directional scattering data
    radi.set_wall_brdf(
        np.arange(len(walls)), brdf, brdf_sources, brdf_receivers)
    # set air absorption
    radi.set_air_attenuation(
        pf.FrequencyData(
            att_np,
            brdf.frequencies))

    # calculate from factors including brdfs
    radi.bake_geometry()
    radi.init_source_energy(source)
    radi.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        etc_time_resolution=time_step,
        etc_duration=duration,
        max_reflection_order=refl_order)

    etc_radiosity = radi.collect_energy_receiver_mono(receivers=receiver)

    return etc_radiosity
