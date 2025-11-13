"""Handler for sta patrizia radiosity simulations."""
import sparrowpy as sp
import pyfar as pf
import numpy as np
import os
import json
import sys
import tqdm

def run(diffuse=True,
        test=True,
        base_dir=os.path.join(os.getcwd(),"..","..",
                              "phd","listening experiment",
                              "synthesis","lib") ):

    # %%  simulation settings
    print("\n\033[93m preparing simulation...\033[00m", end=" ")
    etc_time_resolution = 1/200
    speed_of_sound = 343
    max_refl = 30
    if diffuse:
        brdf_order = 1
    else:
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
    if diffuse:
        mat_filename = "test_materials.json"
    else:
        mat_filename = "acoustic_materials_RISC.json"

    with open(os.path.join(base_dir,
                           mat_filename))as mat_file:
        material_data = json.load(mat_file)

    samples = pf.samplings.sph_gaussian(brdf_order)
    if brdf_order == 1:
        samples.rotate('y',[-90])
    brdf_sources=samples[np.where((samples.elevation*180/np.pi>=0))].copy()
    brdf_receivers=samples[np.where((samples.elevation*180/np.pi>=0))].copy()


    for material_name in material_data.keys():

        if (frequencies!=material_data[material_name]["frequency"]).any():
            raise ValueError("material frequencies do" +
                             "not match input frequency bands")

        s=material_data[material_name]["scattering"]
        a=material_data[material_name]["absorption"]

        if brdf_order == 1:
            a = 1-s*(1-a)
            s = 1.

        brdf = sp.brdf.create_from_scattering(
            brdf_sources,
            brdf_receivers,
            pf.FrequencyData(s, material_data[material_name]["frequency"]),
            pf.FrequencyData(a, material_data[material_name]["frequency"]))

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
    for srcid in tqdm.tqdm(range(source.cshape[0])):
        exchange(srcid, source, receiver, radi,
                    speed_of_sound, etc_time_resolution,
                    max_duration, delays, max_refl, geom_id)
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

    radi.init_source_energy(source[srcID], source_power=2*1e3)

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

    test = False
    diffuse = False
    base_dir=os.path.join(os.getcwd(),"..","..",
                          "phd","listening experiment",
                          "synthesis","lib")

    if len(args)>0:
        test = bool(int(args[0]))
        if len(args)>1:
            diffuse = bool(int(args[1]))
            if len(args)>2:
                base_dir=args[2]

    run(diffuse=diffuse, test=test, base_dir=base_dir)
