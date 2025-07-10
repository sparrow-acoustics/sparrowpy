
"""runs simulation"""
import numpy as np
import sparrowpy as sp
import pyfar as pf

def run_simu(walls, source, receiver,
             patch_size=1, absorption=.1, scattering=1,
             speed_of_sound=343.26, time_step=.1, duration=.5,
             refl_order=3, freq=np.array([1000])):
    # create object
    radi = sp.DirectionalRadiosityFast.from_polygon(walls, patch_size)
    # create directional scattering data (totally diffuse)
    brdf_sources = pf.Coordinates(0, 0, 1, weights=1)
    brdf_receivers = pf.Coordinates(0, 0, 1, weights=1)

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
            np.zeros_like(brdf.frequencies),
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
