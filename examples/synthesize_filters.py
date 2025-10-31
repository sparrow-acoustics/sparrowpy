"""Handler for sta patrizia radiosity simulations."""
import sparrowpy as sp
import pyfar as pf
import numpy as np
import os
from tqdm import tqdm
import sys
from glob import glob


def run(test=True,
        base_dir=os.path.join(os.getcwd(),"..","..",
                              "phd","listening experiment",
                              "synthesis","lib") ):
    # %% settings
    sampling_rate = 44100 # Hz

    # %% load data
    print("\n\033[93m loading etc data...\033[00m", end=" ")

    if test:
        geom_id = "radi_test"
    else:
        geom_id= "radi_work"

    radi = sp.DirectionalRadiosityFast.from_read(os.path.join(base_dir,
                                                              "baked_"+geom_id+".far"))
    etcfiles = glob(os.path.join(base_dir,
                                 "etcs",
                                 "*"+geom_id+"_patchwise_pos"+"*.far"))
    etc_data = pf.io.read(etcfiles[0])
    etc=etc_data["etc"]
    patch_filter = etc_data["patch_filter"]
    duration = etc.times[-1]
    print("\033[92m Done!\033[00m")


    print("\n\033[93m generating and filtering noise signal...\033[00m", end=" ")
    refl_density = pf.TimeData(data = sampling_rate/2*np.ones_like(etc.times),
                               times=etc.times)

    noisefilename = os.path.join(base_dir,"filters","noise.far")

    if not os.path.exists(noisefilename):

        noise = sp.dsp.dirac_sequence(n_samples=int(duration*sampling_rate),
                                    reflection_density=refl_density,
                                    sampling_rate=sampling_rate)

        noise_filtered,bw = sp.dsp.band_filter_signal(signal=noise,
                                                frequencies=np.array(pf.dsp.filter.fractional_octave_frequencies()[0]),
                                                num_fractions=1)

        pf.io.write(filename=noisefilename,
                    noise_filtered=noise_filtered,
                    bw=bw)
    else:

        noisefile = pf.io.read(noisefilename)
        noise_filtered = noisefile["noise_filtered"]
        bw = noisefile["bw"]

    out_filter = pf.Signal(data=np.zeros_like(noise_filtered[0].time),
                           sampling_rate=noise_filtered.sampling_rate)
    print("\033[92m Done!\033[00m")


    print("\n\033[93m loading and filtering hrirs...\033[00m", end=" ")
    hrir,hrir_coords,_=pf.io.read_sofa(os.path.join(base_dir,"FABIAN_HRIR_measured_HATO_0.sofa"))
    hrir_ids = hrirs_per_patch(coords=hrir_coords,
                               patch_positions=radi.patches_center[patch_filter,:])

    hrir = hrir[hrir_ids,:]
    print("\033[92m Done!\033[00m")

    print("\n\033[93m convolving position-wise filters with hrirs...\033[00m",
          end="\n")
    for i, etc_path in enumerate(tqdm(etcfiles)):

        etc = pf.io.read(etc_path)["etc"]

        filter_patchwise = sp.dsp.weight_signal_by_etc(energy_time_curve=etc,
                                                   signal=noise_filtered,
                                                   bandwidth=bw)
        filter_patchwise.time = np.repeat(
            np.sum(filter_patchwise.time, axis=1)[:,np.newaxis,:],2,axis=1)

        out_filter = pf.dsp.convolve(signal1=filter_patchwise, signal2=hrir)

        out_filter.time = np.sum(out_filter.time, axis=0)

        pf.io.write(os.path.join(base_dir,"filters",geom_id+"_filter_"+str(i)+".far"),
                    bin_filter=out_filter,compress=False)
    print("\033[92m Done!\033[00m")


################################################
################################################

def hrirs_per_patch(coords,patch_positions, source_view_azimuth=40.):

    patch_positions /= np.linalg.norm(patch_positions, axis=1)[:,None]
    patch_positions *= coords[0].radius

    idcs = np.empty((patch_positions.shape[0]),dtype=int)

    coords.rotate('z', [source_view_azimuth])

    for i,pos in enumerate(patch_positions):
        sqdiff = np.sum((coords.cartesian-pos)**2, axis=1)
        idcs[i] = np.argmin(sqdiff)

    return np.array(idcs)


######################################
######################################
if __name__ == "__main__":

    args = sys.argv[1:]

    test = True
    base_dir=os.path.join(os.getcwd(),"..","..",
                          "phd","listening experiment",
                          "synthesis","lib")

    if len(args)>0:
        test = bool(int(args[0]))
        if len(args)>1:
            base_dir=args[2]

    run(test=test, base_dir=base_dir)
