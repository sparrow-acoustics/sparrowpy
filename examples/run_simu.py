
"""runs simulation"""
import numpy as np
import sparrowpy as sp
import pyfar as pf
import tracemalloc
import pyrato
import sofar as sf
import time
from reduce_s_d import get_bsc, get_s_rand_from_bsc

def run_simu_mem(walls, source, receiver,
             patch_size=1, absorption=.1, scattering=1,
             speed_of_sound=343.26, time_step=.1, duration=.5,
             refl_order=3, freq=np.array([1000]),
             att=np.array([4.664731873821475/1000]),res=4):

    att=pyrato.air_attenuation_coefficient(freq)
    att_np= att* 0.115129254650564

    # set brdfs
    samples = pf.samplings.sph_gaussian(res)
    pos_hemisphere = np.where((samples.elevation*180/np.pi > 0))
    brdf_sources = samples[pos_hemisphere].copy()
    brdf_receivers = samples[pos_hemisphere].copy()

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
             refl_order=3, freq=np.array([1000]),
             att=np.array([4.664731873821475/1000]),
             res=4):

    t = []
    att_np= att* 0.115129254650564

    # set brdfs
    samples = pf.samplings.sph_gaussian(res)
    pos_hemisphere = np.where((samples.elevation*180/np.pi > 0))
    brdf_sources = samples[pos_hemisphere].copy()
    brdf_receivers = samples[pos_hemisphere].copy()

    t0=time.time()
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
    t.append(time.time()-t0)

    t0=time.time()
    # calculate from factors including brdfs
    radi.bake_geometry()
    t.append(time.time()-t0)

    t0=time.time()
    radi.init_source_energy(source)
    radi.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        etc_time_resolution=time_step,
        etc_duration=duration,
        max_reflection_order=refl_order)
    t.append(time.time()-t0)

    t0=time.time()
    etc_radiosity = radi.collect_energy_receiver_mono(receivers=receiver)
    t.append(time.time()-t0)

    return etc_radiosity,t

def run_simu_pure(
        walls, source, receiver,
        patch_size=2,
        speed_of_sound=343.26, time_step=.002, duration=2.,
        refl_order=50,
        file_wall=None,
        file_ground=None):

    # create object
    radi = sp.DirectionalRadiosityFast.from_polygon(walls, patch_size)

    # read brdfs
    sofa_ground = sf.read_sofa(file_ground)
    brdf_ground, brdf_sources, brdf_receivers = pf.io.convert_sofa(sofa_ground)
    
    sofa_wall = sf.read_sofa(file_wall)
    brdf_walls, brdf_sources, brdf_receivers = pf.io.convert_sofa(sofa_wall)
    brdf_sources.weights = sofa_wall.SourceWeights
    brdf_receivers.weights = sofa_wall.ReceiverWeights

    ground_ind = np.where(np.dot(radi.walls_normal,np.array([0,0,1]))>.9)[0]
    wall_ind = np.where(
        np.abs(np.dot(radi.walls_normal,np.array([0,0,1])))<1e-6)[0]

    # set directional scattering data
    radi.set_wall_brdf(
        ground_ind, brdf_ground, brdf_sources, brdf_receivers)
    radi.set_wall_brdf(
        wall_ind, brdf_walls, brdf_sources, brdf_receivers)

    # calculate from factors including brdfs
    radi.bake_geometry()
    radi.init_source_energy(source)
    radi.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        etc_time_resolution=time_step,
        etc_duration=duration,
        max_reflection_order=refl_order)
    etc_radiosity = radi.collect_energy_receiver_mono(receivers=receiver,
                                                      direct_sound=True)

    return etc_radiosity
