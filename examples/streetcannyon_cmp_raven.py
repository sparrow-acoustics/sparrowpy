# %%
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import pyfar as pf
import pyrato
%matplotlib inline

OUT_DIR = os.path.join(os.getcwd(),"out")

# %%
font={
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Helvetica",
    "font.size": 12,
}



plt.rcParams.update(font)

def create_fig():
    figure,ax = plt.subplots(figsize=(3,2))
    plt.grid()
    return figure, ax

def create_fig2():
    figure,ax = plt.subplots(figsize=(5,3))
    plt.grid()
    return figure, ax

def export_fig(fig, filename,out_dir=OUT_DIR, fformat=".pdf"):
    fig.savefig(os.path.join(out_dir,filename+fformat), bbox_inches='tight')

tlabel="$$rt \\quad [\\mathrm{s}]$$"
mlabel="peak memory [MiB]"
mlegend=["baking", "propagation","collection"]
tlegend=["baking", "propagation","collection","total"]

# %%
print(os.path.join(OUT_DIR, "proof_etcs.far"))
etcs = pf.io.read(os.path.join(OUT_DIR, "proof_etcs.far"))
raven = np.loadtxt(os.path.join(OUT_DIR, 'raven_streetcanyon.csv'), delimiter=",")
# %%
print(raven[0, 3:-2])
raven = pf.TimeData(
    raven[1:, 3:-2].T,
    raven[1:, 0],
)


custom=etcs["custom_etc"]
diff = etcs["diffuse_etc"]

print(etcs["freqs"])
# %%
is_decay = True
for iband in range(5):
    all_etcs =  pf.utils.concatenate_channels([
            custom[0, iband],
            diff[0, iband],
            raven[iband]/(4*np.pi),
        ])
    if is_decay:
        edcs = pyrato.edc.schroeder_integration(all_etcs, True)
    else:
        edcs = all_etcs

    plt.figure()
    pf.plot.time(edcs[0], dB=True, log_prefix=10,
                label="custom",
                linestyle="-")
    pf.plot.time(edcs[1], dB=True, log_prefix=10,
                label="diffuse",
                linestyle="--")
    pf.plot.time(edcs[2]/(4*np.pi), dB=True, log_prefix=10,
                label="Raven",
                linestyle="--")
    plt.legend()
    plt.xlabel("Time  [s]")
    if is_decay:
        plt.ylabel("Energy Decay Curve [dB]")
    else:
        plt.ylabel("Energy Time Curve [dB]")
    plt.ylim([-150,-40])
# %%
