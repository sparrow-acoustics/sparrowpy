#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import matplotlib.pyplot as plt
import numpy as np
import pyfar as pf
import spharpy

import sparrowpy as sp

font = {
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Calibri light",
    "font.size": 16,
}
plt.rcParams.update(font)

base_dir = os.path.join(os.getcwd(), "examples", "resources/")

# In[2]:


# simulation settings
sr_etc = 200
etc_time_resolution = 1 / sr_etc
speed_of_sound = 337
att_dB = 2.7 / 1000
freqcy = 500

# In[3]:


IR_measured = pf.io.read_audio(base_dir + "raw.wav")

latency = int(2300) - int(etc_time_resolution * IR_measured.sampling_rate)

IR_measured = pf.Signal(
    data=IR_measured.time[0, latency:],
    sampling_rate=IR_measured.sampling_rate,
)

print(IR_measured.signal_type)

IR_filtered = pf.dsp.filter.fractional_octave_bands(
    IR_measured,
    3,
    frequency_range=(20, 20000),
)
center_freqs = pf.dsp.filter.fractional_octave_frequencies(3, (20, 20000))


# In[4]:


# generate measured etc from IR data
sr_src = IR_measured.sampling_rate

duration = IR_measured.signal_length
n_etc_samples = int(duration * sr_etc) + 1

etc_array = np.zeros((IR_filtered.cshape[0], n_etc_samples))

for freqbin in range(IR_filtered.cshape[0]):
    src = IR_filtered.time[freqbin, 0, :]
    for i in range(src.shape[0]):
        etc_array[freqbin, int(sr_etc * i / sr_src)] += np.square(src[i])
    etc_level = np.sum(np.square(src[2 : int(0.1 * sr_src)])) / (
        int(0.1 * sr_src) - 2
    )

etc_array = (etc_array - etc_level) * etc_time_resolution

freqbin = np.where(center_freqs[0] == freqcy)

etc_measurement = pf.Signal(
    data=etc_array[freqbin],
    sampling_rate=sr_etc,
    comment="measured ETC",
)


# In[141]:


# radiosity object from file, whole wall-patch shabang
radi = sp.DirectionalRadiosityFast.from_file(
    filepath="examples/resources/reduced_ihtapark2.blend",
    wall_auto_assembly=False,
    geometry_identifier="Geometry",
)

# In[142]:


# set material properties
materials = np.array(
    [
        "ITAs_acoustic_facade1",
        "ITAs_acoustic_facade2",
        "ITAs_acoustic_facade3",
        "ITAs_acoustic_facade4",
        "ITAs_acoustic_facade5",
        "ITAs_acoustic_grass",
        "ITAs_acoustic_asphalt",
    ],
)
alpha = np.empty((materials.shape[0]))
sigma = np.ones((materials.shape[0]))

for i, material in enumerate(materials):
    match material:
        case "ITAs_acoustic_facade1":
            alpha[i] = 0.07
            sigma[i] = 0.61391
        case "ITAs_acoustic_facade2":
            alpha[i] = 0.07
            sigma[i] = 0.63433
        case "ITAs_acoustic_facade3":
            alpha[i] = 0.04
            sigma[i] = 0.03
        case "ITAs_acoustic_facade4":
            alpha[i] = 0.04
            sigma[i] = 0.03
        case "ITAs_acoustic_facade5":
            alpha[i] = 0.04
            sigma[i] = 0.03
        case "ITAs_acoustic_grass":
            alpha[i] = 0.28
            sigma[i] = 0.25
        case "ITAs_acoustic_asphalt":
            alpha[i] = 0.01
            sigma[i] = 0.0


# In[143]:


# set brdfs
samples = spharpy.samplings.gaussian(16)
brdf_sources = samples[np.where((samples.elevation * 180 / np.pi >= 0))].copy()
brdf_receivers = samples[
    np.where((samples.elevation * 180 / np.pi >= 0))
].copy()

brdf_sources.show()


# In[144]:


# set brdfs
frequencies = np.array([freqcy])

# set air absorption
att_Np = att_dB * 0.115129254

radi.set_air_attenuation(
    pf.FrequencyData(att_Np * np.ones_like(frequencies), frequencies),
)

for i, material in enumerate(materials):
    brdf = sp.brdf.create_from_scattering(
        brdf_sources,
        brdf_receivers,
        pf.FrequencyData(sigma[i], frequencies),
        pf.FrequencyData(alpha[i], frequencies),
    )

    # set directional scattering data
    radi.set_wall_brdf(
        np.where(radi._walls_material == material)[0],
        brdf,
        brdf_sources,
        brdf_receivers,
    )


# In[145]:


radi.bake_geometry()


radi.write("cowabunga.far")
# In[146]:


source = pf.Coordinates(0.0254 * 924.05, 0.0254 * -820.39, 0.0254 * 310.546)
# initialize source energy
radi.init_source_energy(source)

# In[147]:


radi.calculate_energy_exchange(
    speed_of_sound=speed_of_sound,
    etc_time_resolution=etc_time_resolution,
    etc_duration=etc_measurement.signal_length,
    max_reflection_order=50,
    recalculate=True,
)


radi.write("cowabunga.far")

# In[148]:


receiver = pf.Coordinates(0.0254 * 2629.9, 0.0254 * 85.71, 0.0254 * -10.967)

etc_radiosity = radi.collect_energy_receiver_mono(
    receivers=receiver,
    direct_sound=True,
)

radi.write("cowabunga.far")

# In[16]:


figure, ax = plt.subplots(figsize=(5, 3))
pf.plot.time(
    signal=etc_measurement,
    dB=True,
    log_prefix=10,
    ax=ax,
    label="measured ETC",
)
plt.legend(fontsize=14)
plt.ylim([-125, -40])
ax.set_ylabel("Magnitude [dB]")
ax.set_xlabel("Time [s]")
plt.grid()
plt.title(f"IHTA Park ETC comparison [{freqcy}Hz]")
figure.savefig(
    os.path.join(os.getcwd(), "figures", "ETCs_ref.svg"),
    bbox_inches="tight",
)
plt.show()


figure, ax = plt.subplots(figsize=(5, 3))
pf.plot.time(
    signal=etc_measurement,
    dB=True,
    log_prefix=10,
    ax=ax,
    label="measured ETC",
)
pf.plot.time(
    signal=etc_radiosity[0],
    dB=True,
    log_prefix=10,
    ax=ax,
    linestyle="--",
    label="simulated ETC",
)
ax.set_ylabel("Magnitude [dB]")
ax.set_xlabel("Time [s]")
plt.grid()
plt.legend(fontsize=14)

figure.savefig(
    os.path.join(os.getcwd(), "figures", "ETCs_vs.svg"),
    bbox_inches="tight",
)
plt.show()
