"""Handler for sta patrizia radiosity simulations."""
import sparrowpy as sp
import pyfar as pf
import numpy as np
import os
import json
import sys
import threading
import time


def run(mono=True,
        test=True,
        base_dir=os.path.join(os.getcwd(),"..","..",
                              "phd","listening experiment",
                              "synthesis","lib") ):

    # %%  simulation settings
    print("\n\033[93m preparing simulation...\033[00m", end=" ")
    etc_time_resolution = 2e-2
    speed_of_sound = 343
    max_refl = 2
    brdf_order = 10
    freqbands = pf.dsp.filter.fractional_octave_frequencies(num_fractions=1) # octave bands
    frequencies = freqbands[0]

    # %% trajectory data loading
    with open(os.path.join(base_dir,
                        "Scenario_86",
                        "S86_trajectory.json"))as traj_file:
            trajectory_data = json.load(traj_file)

    source_positions = np.array(trajectory_data["v_trajectory"])
    delays = np.array(trajectory_data["v_duration"])

    source_positions = source_positions.T

    max_duration = np.max(delays) + 3

    if mono:
        source = pf.Coordinates(0,0,0)
        receiver = pf.Coordinates(source_positions[0],
                                   source_positions[1],
                                   source_positions[2])
    else:
        receiver = pf.Coordinates(0,0,0)
        source = pf.Coordinates(source_positions[0],
                                   source_positions[1],
                                   source_positions[2])

    # %% load geometry
    if test:
        geom_id = "radi_test"
    else:
        geom_id= "radi_work"

    radi = sp.DirectionalRadiosityFast.from_file(
                    filepath=os.path.join(base_dir,"SPatrizia_final.blend"),
                    wall_auto_assembly=False,
                    geometry_identifier=geom_id,
                    )

    # %% load materials and assign BRDFs, air attenuation
    with open(os.path.join(base_dir,
                         "acoustic_materials_RISC.json"))as mat_file:
        material_data = json.load(mat_file)

    samples = pf.samplings.sph_gaussian(brdf_order)
    brdf_sources=samples[np.where((samples.elevation*180/np.pi>=0))].copy()
    brdf_receivers=samples[np.where((samples.elevation*180/np.pi>=0))].copy()


    for material_name in material_data.keys():

        if (frequencies!=material_data[material_name]["frequency"]).any():
            raise ValueError("material frequencies do" +
                             "not match input frequency bands")

        brdf = sp.brdf.create_from_scattering(
            brdf_sources,
            brdf_receivers,
            pf.FrequencyData(material_data[material_name]["scattering"],
                            material_data[material_name]["frequency"]),
            pf.FrequencyData(material_data[material_name]["absorption"],
                            material_data[material_name]["frequency"]))

        # set directional scattering data
        radi.set_wall_brdf(
            np.where(radi._walls_material==material_name)[0],
            brdf,
            brdf_sources,
            brdf_receivers)

    radi.set_air_attenuation(
    pf.FrequencyData(
        trajectory_data["v_attenuation"],
        frequencies))

    print("\033[92m Done!\033[00m")

    # %% bake geometry or load baked geometry
    print("\n\033[93m baking geometry...\033[00m", end=" ")
    baked_filename=os.path.join(base_dir,"baked_"+geom_id+".far")
    if not os.path.exists(baked_filename):
        radi.bake_geometry()
        radi.write(baked_filename)
    else:
        radi = sp.DirectionalRadiosityFast.from_read(baked_filename)

    print("\033[92m Done!\033[00m")

    # %% energy exchange
    print("\033[93m exchanging energy...\033[00m", end="\n")
    if mono:
        radi.init_source_energy(source)

        radi.calculate_energy_exchange(
                speed_of_sound=speed_of_sound,
                etc_time_resolution=etc_time_resolution,
                etc_duration=max_duration,
                max_reflection_order=max_refl,
                recalculate=True)

        for direct_sound in [True,False]:
            etc = radi.collect_energy_receiver_mono(
                receivers=receiver,direct_sound=direct_sound)

            if direct_sound:
                mononame = geom_id+"_mono_WITH_direct.far"
            else:
                mononame = geom_id+"_mono_NO_direct.far"

            pf.io.write(os.path.join(base_dir,"etcs",mononame),
                        etc=etc,
                        compress=True)
    else:

        t0 = time.time()
        threads = []
        for srcID in range(source.cshape[0]):
            t = threading.Thread(target=exchange,
                             args=(srcID, source, receiver, radi,
                                   speed_of_sound, etc_time_resolution,
                                   max_duration, delays, max_refl, geom_id))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        tthreads = time.time()-t0

        t0 = time.time()
        for srcID in range(source.cshape[0]):
            exchange(srcID, source, receiver, radi,
                     speed_of_sound, etc_time_resolution,
                     max_duration, delays, max_refl, geom_id)
        tsimple = time.time()-t0

    print(f'simple: {tsimple}s;   multithread: {tthreads}s.')

    print("\033[92m Done!\033[00m")

def filter_patches(etc):

    patch_filter = []

    for patchID in range(etc.cshape[1]):
        if not (etc[0,patchID].time==0).all():
            patch_filter.append(patchID)

    return np.array(patch_filter)

def exchange(srcID, source, receiver, radi, speed_of_sound,
             etc_time_resolution, max_duration,
             delays,
             max_refl, geom_id):

    radi.init_source_energy(source[srcID], source_power=1e10)

    radi.calculate_energy_exchange(
            speed_of_sound=speed_of_sound,
            etc_time_resolution=etc_time_resolution,
            etc_duration=max_duration,
            etc_clip=delays[srcID],
            max_reflection_order=max_refl,
            recalculate=True)

    etc = radi.collect_energy_receiver_patchwise(receivers=receiver,
                                                    etc_clip=delays[srcID])

    patch_filter = filter_patches(etc)

    if not len(patch_filter)==0:
        etcname=geom_id+"_patchwise_pos"+str(srcID)+".far"

        pf.io.write(os.path.join(base_dir,"etcs",etcname),
                    etc=etc[0,patch_filter],
                    patch_filter=patch_filter,
                    compress=True)

if __name__ == "__main__":

    args = sys.argv[1:]

    test = True
    mono = False
    base_dir=os.path.join(os.getcwd(),"..","..",
                          "phd","listening experiment",
                          "synthesis","lib")

    if len(args)>0:
        test = bool(int(args[0]))
        if len(args)>1:
            mono = bool(int(args[1]))
            if len(args)>2:
                base_dir=args[2]

    run(mono=mono, test=test, base_dir=base_dir)
