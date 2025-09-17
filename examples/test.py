import pyfar as pf
import sparrowpy as sp
import numpy as np
import matplotlib.pyplot as plt
n_samples = 44100
white_noise = pf.dsp.normalize(pf.signals.noise(n_samples,rms=1))

nom_frequencies,_,cutoffs = pf.dsp.filter.fractional_octave_frequencies(
    return_cutoff=True,
    )
n_frequencies = cutoffs[0].shape[0]
bandwidth=cutoffs[1]-cutoffs[0]

noise_bandwise = pf.dsp.filter.fractional_octave_bands(
    signal=white_noise,
    num_fractions=1,
)

noise_bandwise.time = np.squeeze(noise_bandwise.time)

delta_t = 1/1000
times = np.arange(0,white_noise.times[-1],delta_t)
decay = np.empty((n_frequencies,times.shape[0]))
for i in range(n_frequencies):
    decay[i,:] = np.exp(-i*times)

etc = pf.TimeData(data=decay,times=times)
pf.plot.time(etc)
plt.show()

weighted_noise_bandwise = sp.dsp.weight_filters_by_etc(
    etc=etc,
    signal=noise_bandwise,
    bandwidth=bandwidth,
)

ax=pf.plot.time(weighted_noise_bandwise,
             label=[f"{nom_frequencies[i]}Hz octave band" \
                            for i in range(nom_frequencies.shape[0])])
ax.legend()
ax.set_title("Frequency-wise decay")
plt.show()




