# %% Histogram generation

from types import NoneType
import numpy as np
import pyfar as pf
from pyfar.dsp.filter import high_shelf  # type: ignore
import sparrowpy as sp
import matplotlib.pyplot as plt
from datetime import datetime
# ruff: noqa: D100 ERA001

def run_energy_simulation(
    dimensions, patch_size, ir_length_s, sampling_rate, max_order_k, source, receiver,
    ) -> np.ndarray:
    """
    Calculate the histogram for a room with dimensions.

    Parameters
    ----------
    dimensions : ndarray[float(#3)]
        Dimensions of the room.
    patch_size : float
        Size of the patches.
    ir_length_s : float
        Length of the histogram.
    sampling_rate : float
        Sampling rate of the original histogram.
    max_order_k : int
        Order of energy exchanges.
    source : pf.Coordinates
        Position of the source.
    receiver : pf.Coordinates
        Position of the receiver.

    Returns
    -------
    histogram : NDArray[n_receivers, n_patches, n_(freq)bins, E_matrix_total '-1'shape]
        Histogram of the room.
    """

    speed_of_sound = 343

    walls = sp.testing.shoebox_room_stub(dimensions[0], dimensions[1], dimensions[2])

    radi = sp.DRadiosityFast.from_polygon(walls, patch_size)

    # maybe problems with the brdf directions for different norm vectors
    brdf = sp.brdf.create_from_scattering(
        source_directions=pf.Coordinates(0, 0, 1, weights=1),
        receiver_directions=pf.Coordinates(0, 0, 1, weights=1),
        scattering_coefficient=pf.FrequencyData(1, [1000]),
        absorption_coefficient=pf.FrequencyData(0, [1000]),
    )

    radi.set_wall_scattering(
        wall_indexes=np.arange(len(walls)).tolist(),
        scattering=brdf,
        sources=pf.Coordinates(0, 0, 1, weights=1),
        receivers=pf.Coordinates(0, 0, 1, weights=1),
    )

    radi.set_air_attenuation(
        pf.FrequencyData([0.2], [1000]),
    )

    radi.set_wall_absorption(
        np.arange(len(walls)).tolist(),
        pf.FrequencyData(np.zeros_like(brdf.frequencies), brdf.frequencies),
    )

    radi.bake_geometry()

    radi.init_source_energy(source)

    radi.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        histogram_time_resolution=1 / sampling_rate,
        histogram_length=ir_length_s,
        max_depth=max_order_k,
    )

    histogram = radi.collect_receiver_energy(
        receiver_pos=receiver,
        speed_of_sound=speed_of_sound,
        histogram_time_resolution=1 / sampling_rate,
        propagation_fx=True,
    )

    return histogram


# %% IR generation
def run_ir_generation(
    hist_sum, room_volume, sampling_rate, delta_rHist, keep_original_hist=False) -> pf.Signal:
    """
    Calculate the impulse response for a histogram.

    Parameters
    ----------
    hist_sum : NDArray.shape()
        Histogram of a simulation.
    room_volume : float/int
        Volume through X*Y*Z of the room.
    sampling_rate : float
        Sampling rate of the original histogram.
    delta_rHist : float
        Time delta for the (reduced) histogram for the IR.
    keep_original_hist : bool
        Reduction of the histogram resolution.

    Returns
    -------
    ir : pf.Signal
        Dirac impulse response of the room.
    """

    speed_of_sound = 343
    sampling_rate_diracS = 48000
    divide_BW = False

    if not keep_original_hist and not delta_rHist < 1/sampling_rate:
        factor_delta_t = delta_rHist * sampling_rate
        print(f"Reduction in histogram resolution: {factor_delta_t}")
        hist_reduced = []
        index_values = range(
            0, len(hist_sum), max(int(factor_delta_t), 1))
        for ix in index_values:
            hist_reduced.append(sum(hist_sum[ix : ix + max(int(factor_delta_t), 1)]))
        time_values = np.arange(0, len(hist_reduced) * delta_rHist, delta_rHist)

        hist_reduced_sig = pf.Signal(hist_reduced, 1/delta_rHist)
        plt.figure()
        pf.plot.time(
            hist_reduced_sig, dB=True, log_prefix=10,
            label=f"Histogram reduced of {room_volume} m^3 room" +
                    f"\nReduction of resolution by {factor_delta_t}x")
        plt.scatter(
            time_values,
            np.multiply(10, np.log10(np.asarray(hist_reduced))),
            label="Separate values of reduced histogram",
        )
        plt.ylim(-200, 0)
        plt.xlabel("seconds")
        plt.legend()
        plt.show()
    else:
        if delta_rHist < 1/sampling_rate:
            print("Delta_rHist is smaller than the original sampling rate. Not relevant!")
        else:
            print("No reduction in histogram resolution!")
        delta_rHist = 1/sampling_rate
        factor_delta_t = 1
        hist_reduced = hist_sum

    # noise sequence with poisson distribution
    rng = np.random.default_rng()
    diracNonZeros = []
    µ_t = 4 * np.pi * pow(speed_of_sound, 3) / room_volume / 1000 ##trash for testing

    time_start = 1/max(sampling_rate_diracS,1000)      # max ~0.3m sound travel time
    time_stop = len(hist_reduced) * delta_rHist     # max ir_length_s
    time_step = 1/sampling_rate_diracS
    for time in np.arange(time_start, time_stop, time_step):
        time_for_itr = time
        while (delta :=\
            (1/(min(µ_t * pow(time, 2), 10000)) * np.log(1/rng.uniform(1e-10, 1)))) <\
                time_for_itr+time_step-time:
            time += delta
            diracNonZeros.append(rng.choice([-time, time], p=[0.5,0.5]))
            ###now random choice of +- dirac but density still poisson distributed formula
            # if time%(1/sampling_rate_diracS) < 1/sampling_rate_diracS/2:
            #     diracNonZeros.append(time)
            # else:
            #     diracNonZeros.append(-time)
            ##time% very sensitive for dirac +- value and bc of sampling rate
    print(f"Sampling rate dirac: {sampling_rate_diracS}\nDelta_rHist: {delta_rHist}\nFactor delta_t: {factor_delta_t}")

    diracS_time = np.arange(0, time_stop, time_step)
    diracS_value = np.zeros_like(diracS_time)
    for time in diracNonZeros:
        ix = int(abs(time)/time_step)
        if diracS_value[ix] == 0:
            diracS_value[ix] = np.sign(time)


    # plot the dirac sequence in time and frequency domain
    plt.figure()
    plt.plot(diracS_time, diracS_value, label="Dirac sequence")
    plt.xlim(0.2, 0.3) #for density checking
    plt.xlabel("seconds")
    plt.ylabel("amplitude")
    #plt.savefig(fname=f"sim_data/f_dirac_{room_volume}m^3_{delta_rHist}s.svg")
    plt.show()

    dirac_sig = pf.Signal(diracS_value, sampling_rate_diracS)
                                        # future other smp_rate
    pf.plot.freq(dirac_sig, label="Dirac sequence freq")
    #plt.xlim(5, 24000)
    plt.show()

    # IEC 61260:1:2014 standard or in the future e.g.Raised Cosine Filter
    filtered_sig, filter_frequencies = pf.dsp.filter.reconstructing_fractional_octave_bands(
        signal=None,
        num_fractions=1,
        frequency_range=(125, 16000),
        overlap=1,
        slope=0,
        n_samples=4096,
        sampling_rate=sampling_rate_diracS,
    )
    a = filtered_sig.process(dirac_sig)
    
    pf.plot.freq(a, label="dirac sequence")
    plt.show()
    if divide_BW:   #placeholder
        a = -1
        #diracS_filtered = filtered_sig.process(dirac_sig)
    else:
        a = -1
        #diracS_filtered = dirac_sig

    diracS_weighted = np.zeros_like(diracS_value)
    factor_s = sampling_rate_diracS*delta_rHist # >1!
    for ix in range(len(hist_reduced)):
        low = int(ix*factor_s)
        high = int((ix+1)*factor_s)
        print(f"low: {low}, high: {high}")
        div = sum(diracS_value[low:high])
        if div == 0:
            div = 1e-10
        diracS_weighted[low:high] =\
            diracS_value[low:high] *\
                np.sqrt(hist_reduced[ix]/pow(div, 2))
            #*np.sqrt(diracS_filtered.frequencyBW/(sampling_rate_diracS/2)))
    #if divide_BW: sum over frequency bands
    sig = pf.Signal(diracS_weighted, sampling_rate_diracS)
    plt.figure()
    pf.plot.time(sig, dB=True, label="Dirac sequence weighted", log_prefix=10)
    #plt.xlim(0, 1) #for check
    plt.xlabel("seconds")
    plt.ylabel("amplitude")
    #plt.savefig(fname=f"sim_data/f_dirac_weighted_{room_volume}m^3_{delta_rHist}s.svg")
    plt.show()

    return pf.Signal(diracS_weighted, sampling_rate_diracS)


# %% Run the functions
update_hist = True

X = 4
Y = 5
Z = 3
room_volume = X * Y * Z

ir_length_s = 2         # default 2
sampling_rate = 1000    # 1/delta_t per second
patch_size = 1.2
max_order_k = 500       # default >200 for probably clean-200dB in the histogram
print(f"X: {X}\nY: {Y}\nZ: {Z}\nPatch size: {patch_size}")

delta_rHist = 0.01           # default 0.1 for the IR

if update_hist:
    start = datetime.now()
    hist_full : np.ndarray = run_energy_simulation(
        [X, Y, Z],
        patch_size,
        ir_length_s,
        sampling_rate,
        max_order_k,
        source=pf.Coordinates(1, 1, 1),
        receiver=pf.Coordinates(2.5, 3.5, 1.6),
    )
    txt_data = np.concatenate(
        (
            np.arange(
                start=0, stop=hist_full.shape[3] / sampling_rate, step=1 / sampling_rate),
            hist_full[0, :, 0, :].reshape(-1),  # one freq bin
        ),
    )
    txt_data = np.reshape(txt_data, (hist_full.shape[1] + 1, hist_full.shape[3]))
    np.savetxt(f"sim_data/f_hist_room_{X}_{Y}_{Z}_{patch_size}.csv", txt_data, delimiter=",")
    delta = datetime.now() - start
    print(f"Time elapsed: {delta}")
else:
    txt_data = np.genfromtxt(
        f"sim_data/f_hist_room_{X}_{Y}_{Z}_{patch_size}.csv",
        delimiter=",",
    )

hist_sum = np.sum(txt_data[1:, :], axis=0)

hist_sum_sig = pf.Signal(hist_sum, sampling_rate)

plt.figure()
pf.plot.time(
    hist_sum_sig, dB=True, log_prefix=10,
    label=f"Histogram of room with size {X}x{Y}x{Z} m\nPatch size of {patch_size}")
plt.ylim(-200, 0)
plt.xlabel("seconds")
plt.legend()
if update_hist:
    plt.savefig(fname=f"sim_data/f_hist_room_{X}_{Y}_{Z}_{patch_size}.svg")
plt.show()

check_what_pf_sig = run_ir_generation(  #return check?????
    hist_sum, room_volume, sampling_rate, delta_rHist, keep_original_hist=True)


# %% noise tests
filterFI, filter_frequencies = pf.dsp.filter.reconstructing_fractional_octave_bands(
    signal=None,##
    num_fractions=1,
    frequency_range=(125, 16000),
    overlap=1,
    slope=0,
    n_samples=4096,
    sampling_rate=48000,
)
y_sum = pf.Signal(np.sum(filterFI.time, 0), filtered_sig.sampling_rate)
pf.plot.freq(filtered_sig, label="Filtered Signal")
plt.show()

# noise sequence with poisson distribution
    rng = np.random.default_rng()

    diracS = []
    µ_t = 4 * np.pi * pow(speed_of_sound, 3) / room_volume

    time = 1/max(sampling_rate,1000) # max 0.3m sound travel time
    while time < len(hist_reduced) * delta_t:   # max ir_length_s
        µ = min(µ_t * pow(time, 2), 10000)

        time_add = 1/µ * np.log(1/rng.uniform(1e-10, 1))
        time += time_add
        if time%(1/sampling_rate) < 1/sampling_rate/2:
            diracS.append(time)
        else:
            diracS.append(-time)
    #time% very sensitive for dirac +- value




a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sampling_rate = 10
factor_delta_t = 1
# noise sequence with poisson distribution
rng = np.random.default_rng()
dirac_sequence = rng.poisson(lam=1, size=len(a))
for ix in range(len(dirac_sequence)):
    if dirac_sequence[ix] != 0:
        dirac_sequence[ix] = rng.choice([-1, 1], p=[0.5,0.5])
plt.figure()
plt.plot(
    np.arange(len(a)), #np.arange(len(hist_reduced) / sampling_rate * factor_delta_t)
    dirac_sequence,
    label="dirac sequence",
)
plt.xlabel("seconds")
plt.ylabel("amplitude")
plt.show()

# spectrum of the poisson diracs
dirac_sig = pf.Signal(
    dirac_sequence,
    sampling_rate/factor_delta_t,
)
#dirac_freq = dirac_sig.freq
plt.plot(
    dirac_sig.freq,
    label="dirac sequence",
)
plt.show()

# e.g Raised Cosine Filter in the future! now IEC 61260:1:2014 standard
filtered_sig, freq_center = pf.dsp.filter.reconstructing_fractional_octave_bands(
    dirac_sig,
    num_fractions=1,
    frequency_range=(125, 16000),
    overlap=1,
    slope=0,
    n_samples=4096,
    sampling_rate=None,
)
pf.plot.freq(filtered_sig, label="dirac sequence")
plt.show()
