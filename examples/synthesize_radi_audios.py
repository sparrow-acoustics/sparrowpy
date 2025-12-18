"""Handler for sta patrizia radiosity simulations."""
import pyfar as pf
import numpy as np
import os
from tqdm import tqdm
import sys
from glob import glob
import re



def run(test=False,
        source_signal=pf.signals.files.guitar(),
        base_dir=os.path.join(os.getcwd(),"lib"),
        filter_id=None):
    # %% settings
    sampling_rate = 48000 # Hz

    # %% load data
    print("\n\033[93m loading filters...\033[00m", end=" ")

    if test:
        geom_id = "radi_test"
    else:
        geom_id= "radi_work"


    if source_signal.sampling_rate!=sampling_rate:
        source_signal = pf.dsp.resample(signal=source_signal,
                                        sampling_rate=sampling_rate)
    if source_signal.cshape != (2,):
        source_signal.time = np.repeat(source_signal.time,2,axis=0)

    if filter_id is not None:
        filterfiles = glob(os.path.join(base_dir,
                                    "filters",
                                    "*"+geom_id+"_filter_"+f"{filter_id:.0}"+"*.far"))
    else:

        filterfiles = glob(os.path.join(base_dir,
                                    "filters",
                                    "*"+geom_id+"_filter_"+"*.far"))

    for ffile in tqdm(filterfiles):

        id = re.search("(?<=filter_).*?(?=.far)",ffile).group()

        if os.path.isfile(ffile):

            bin_filter = pf.io.read(filename=ffile)["bin_filter"]

        out_signal = pf.dsp.convolve(signal1=source_signal,
                                     signal2=bin_filter,mode='full')

        out_signal.time = out_signal.time[...,int(.5*out_signal.sampling_rate):]

        pf.io.write_audio(signal=out_signal,
                          filename=os.path.join(base_dir,
                                                "audio",
                                                source_signal.comment+
                                                f"_{id}.wav"),subtype='DOUBLE')

        print(ffile)
        print(id)
        print()

################################################
################################################


######################################
######################################
if __name__ == "__main__":

    args = sys.argv[1:]

    test = True
    base_dir=os.path.join(os.getcwd(),"..","..",
                          "phd","listening_experiment",
                          "synthesis","lib")

    if len(args)>0:
        test = bool(int(args[0]))
        if len(args)>1:
            base_dir=args[1]

    run(test=test, base_dir=base_dir)
