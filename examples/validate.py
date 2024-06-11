# %%
"""Test the radiosity.Radiosity module."""
import numpy as np
import pyfar as pf
import sparapy as sp
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
X = 5
Y = 6
Z = 4
patch_size = 0.5
ir_length_s = 2
sampling_rate = 1000
max_order_k = 150
speed_of_sound = 343

absorption = 0.1
S = (2*X*Y) + (2*X*Z) + (2*Y*Z)
A = S*absorption
alpha_dash = A/S
r_h = 1/4*np.sqrt(A/np.pi)
print(f'reverberation distance is {r_h}m')
V = X*Y*Z
RT = 24*np.log(10)/(speed_of_sound)*V/(-S*np.log(1-alpha_dash))
print(f'reverberation time is {RT}s')
# create geometry
walls = sp.testing.shoebox_room_stub(X, Y, Z)

radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(
    walls, patch_size)
# create directivity

gaussian = pf.samplings.sph_gaussian(sh_order=1)
gaussian = gaussian[gaussian.z>0]
sources = gaussian.copy()
receivers = gaussian.copy()
frequencies = pf.dsp.filter.fractional_octave_frequencies(
    1, (100, 1000))[0]
data = np.ones((sources.csize, receivers.csize, frequencies.size))
data = pf.FrequencyData(data, frequencies)

radiosity.set_wall_scattering(
    np.arange(6), data, sources, receivers)
radiosity.set_air_attenuation(
    pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
radiosity.set_wall_absorption(
    np.arange(6),
    pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))



# %%

# run simulation
radiosity.check_visibility()
radiosity.calculate_form_factors()
radiosity.calculate_form_factors_directivity()
radiosity.calculate_energy_exchange(max_order_k)

# %%
# test energy at receiver
receiver_pos = [
    [2, 3, 2], [3, 2, 2], [3, 3, 2], [3, 4, 2],
    [2, 3, 2.5], [3, 2, 2.5], [3, 3, 2.5], [3, 4, 2.5],
    [2, 2, 1.5]
    ]
histograms = []
source_pos = np.array([2, 2, 2])

radiosity.init_energy(source_pos)

for pos in receiver_pos:
    receiver = sp.geometry.Receiver(pos, [0, 1, 0], [0, 0, 1])
    histograms.append(radiosity.collect_energy_receiver(
        receiver_pos, histogram_time_resolution=1e-3, histogram_time_length=1))
irs_new = np.array(histograms).squeeze()
reverberation = pf.Signal(irs_new, sampling_rate=sampling_rate)


# %%
direct_sound_list = []
direct_analytic = []
for pos in receiver_pos:
    r = np.sqrt(np.sum((np.array(pos)-source_pos)**2))
    direct_sound = (1/(4 * np.pi * np.square(r)))
    delay_dir = int(r/speed_of_sound*sampling_rate)
    direct_analytic.append(direct_sound)
    direct_sound_list.append(pf.signals.impulse(
        reverberation.n_samples, delay_dir, direct_sound, sampling_rate).time)
direct_analytic = np.array(direct_analytic)
direct_sound = pf.Signal(
    np.array(direct_sound_list).squeeze(), sampling_rate=sampling_rate)
result = pf.utils.concatenate_channels([reverberation, direct_sound])

# %%
E_direct = np.sum(direct_sound.time, axis=-1)
E_reverb = np.sum(reverberation.time, axis=-1)

E_ratio = E_reverb/E_direct

E_direct_analytical = 1/(4*np.pi*r**2)
E_reverb_analytical = 4/A
E_ratio_analytical = E_reverb_analytical/E_direct_analytical

t = reverberation.times
w_0 = E_reverb_analytical/ V # Kuttruff Eq 4.7
t_0 = 0.03
reverberation_analytic = w_0 * np.exp(+(
    speed_of_sound*S*np.log(1-alpha_dash)/(4*V))*(t-t_0)) # Kuttruff Eq 4.10
reverberation_analytic = pf.Signal(reverberation_analytic, sampling_rate=sampling_rate)
plt.figure()
pf.plot.time(
    reverberation_analytic, dB=True, log_prefix=10,
    label=f'analytical E_rev={E_reverb_analytical:0.2f}', color='r')
for i in range(len(receiver_pos)):
    e_rel = (E_reverb[i]/E_reverb_analytical)
    pf.plot.time(
        reverberation[i], dB=True, log_prefix=10,
        label=f'simulated E_rev={E_reverb[i]:0.2f} ratio={e_rel:0.2f}',
        linestyle='--')
plt.legend()
plt.show()

# %%
