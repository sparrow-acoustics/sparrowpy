# %% Histogram generation
import numpy as np
import pyfar as pf  # type: ignore
import sparrowpy as sp
import matplotlib.pyplot as plt
from datetime import datetime
# print(pf.plot.shortcuts())
# ruff: noqa: D100 ERA001 W605

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

    print("Debug: Start radiosity")
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
    print("Debug: wait bake geometry")
    radi.bake_geometry()
    print("Debug: wait init energy")
    radi.init_source_energy(source)
    print("Debug: wait exchange")
    radi.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        histogram_time_resolution=1 / sampling_rate,
        histogram_length=ir_length_s,
        max_depth=max_order_k,
    )
    print("Debug: wait collect")
    histogram = radi.collect_receiver_energy(
        receiver_pos=receiver,
        speed_of_sound=speed_of_sound,
        histogram_time_resolution=1 / sampling_rate,
        propagation_fx=True,
    )
    print("Debug: returning")
    return histogram


# %% Histogram comparison
run_independent = False     # check end of cell for exec

def run_csv_comparison(f1_string, sampling_rate1, f2_string, sampling_rate2,
                       res_reduction_data2 = False, conv_to_energy_data2 = False):
    """
    Compare two histograms from csv file datasets.

    Parameters
    ----------
    f1_string : string
        File 1 path and filename.
    sampling_rate1 : int
        Sampling rate of file 1.
    f2_string : string
        File 2 path and filename.
    sampling_rate2 : int
        Sampling rate of file 2.
    res_reduction_data2 : bool
        Reduction of the resolution of file 2.
    conv_to_energy_data2 : bool
        IR to histogram of file 2 for comparison to original histogram.
    """

    # input and sum of patches
    txt_data1 = np.genfromtxt(f1_string, delimiter=",")
    txt_data2 = np.genfromtxt(f2_string, delimiter=",")

    # catch previously summed data saved in file (like in IR case) no patch info
    #print(txt_data2.shape, txt_data2.shape[0])
    #if txt_data1.shape[0] > 2:
    txt_data1 = np.sum(txt_data1[1:, :], axis=0)
    #if txt_data2.shape[0] > 2:
    txt_data2 = np.sum(txt_data2[1:, :], axis=0)


    if conv_to_energy_data2:
        txt_data2 = np.array([txt_data2[i] ** 2 for i in range(len(txt_data2))])
        print("Data of File 2 of IR is converted back to energy")


    if res_reduction_data2:
        print(f"Length of Signal 1: {txt_data1.shape[0]/sampling_rate1}s")
        print(f"Length of Signal 2: {txt_data2.shape[0]/sampling_rate2}s" +
            "\n\tif mismatch -> adjust!")

        factor_samRate = int(sampling_rate2 / sampling_rate1)
        print(f"Factor sampling rate: {factor_samRate}")
        txt_dataReducedRate = []
        for ix in range(int(len(txt_data2) / factor_samRate)):
            txt_dataReducedRate.append(
                sum(txt_data2[ix*factor_samRate : ix*factor_samRate + factor_samRate]))
        txt_data2 = np.asarray(txt_dataReducedRate)

    # plotting the histograms with equal sampling rates
    plt.figure()
    pf.plot.time(pf.Signal(txt_data1, sampling_rate1), dB=True)
    pf.plot.time(pf.Signal(txt_data2, sampling_rate2), dB=True)
    plt.show()
    # plot the comparison
    pf.plot.time(pf.Signal(txt_data2, sampling_rate1))
    txt_data_diff = np.array([txt_data1[i] / txt_data2[i] - 1 if txt_data2[i] else 0
                            for i in range(len(txt_data2))]) # txt_data equal length
    txt_data_diff_sig = pf.Signal(txt_data_diff, sampling_rate1)
    plt.figure()
    pf.plot.time(txt_data_diff_sig, dB=True, log_prefix=10, label="Histogram comparison")
    plt.xlabel("seconds")
    #plt.xlim(0, 0.05)
    plt.ylim(-50, 100)
    plt.legend()
    plt.text(0.15, 0, "Summed energy difference:   " +
            f"${round(10*np.log10(sum(txt_data_diff[:])), 2)}\,dB$", fontsize=12)
    plt.ion()
    plt.show()
    print(f"Summed energy difference:\n\t{round(sum(txt_data_diff[:]), 3)}" +
        f" or {round(10*np.log10(sum(txt_data_diff[:])), 2)} dB")
    print("Maximum single value difference: " +
        f"{round(10*np.log10(max(txt_data_diff[:])), 2)} dB")

if run_independent:
    file1 = "sim_data/f_hist_room_4_5_3_1_8k.csv"
    file2 = "sim_data/f_ir_room_.csv"
    run_csv_comparison(file1, 8000, file2, 48000,
                            res_reduction_data2=True, conv_to_energy_data2=True)


# %% IR generation
def run_ir_generation(
    hist_sum, room_volume, sampling_rate, keep_original_hist=True,
    delta_reducHist=None) -> pf.Signal:
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
    keep_original_hist : bool
        Reduction of the histogram resolution.
    delta_reducHist : float
        Time delta for the (reduced) histogram for the IR.

    Returns
    -------
    ir : pf.Signal
        Dirac impulse response of the room.
    """

    delta_reducHist = 1e-10 if delta_reducHist is None else delta_reducHist
    speed_of_sound = 343
    sampling_rate_dirac = 48000
    divide_BW = True

    if not keep_original_hist and not delta_reducHist < 1/sampling_rate:
        factor_delta = delta_reducHist * sampling_rate
        print(f"Reduction in histogram resolution: {factor_delta}")
        hist_reduced = []
        index_values = range(
            0, len(hist_sum), int(factor_delta))
        for ix in index_values:
            hist_reduced.append(sum(hist_sum[ix : ix + int(factor_delta)]))
        time_values = np.arange(0, len(hist_reduced) * delta_reducHist, delta_reducHist)

        hist_reduced_sig = pf.Signal(hist_reduced, 1/delta_reducHist)
        pf.plot.time(
            hist_reduced_sig, dB=True, log_prefix=10,
            label=f"Histogram reduced of {room_volume} m^3 room" +
                    f"\nReduction of resolution by {factor_delta}x")
        plt.scatter(
            time_values,
            np.multiply(10, np.log10(np.asarray(hist_reduced))),
            label="Separate values of reduced histogram")
        plt.ylim(-200, 0)
        plt.xlabel("seconds")
        plt.legend()
        plt.show()
    else:
        if not keep_original_hist and delta_reducHist < 1 / sampling_rate:
            print(
                f"Delta_t for the reduced histogram of {delta_reducHist} is smaller\n"+
                "than the original histogram resolution given by the sampling rate\n"+
                f"at {1/sampling_rate}.\n! Incorrect call !\n"+
                "Using original histogram (resolution).")
        else:
            print("Histogram resolution not reduced.")
        delta_reducHist = 1/sampling_rate
        factor_delta = 1
        hist_reduced = hist_sum
        hist_reduced_sig = pf.Signal(hist_reduced, 1 / delta_reducHist)

    # noise sequence with poisson distribution
    rng = np.random.default_rng()
    diracNonZeros = []
    µ_t = 4 * np.pi * pow(speed_of_sound, 3) / room_volume ##div by 1000 for testing

    time_start = 1/max(sampling_rate_dirac,1000)      # max ~0.3m sound travel time
    time_stop = len(hist_reduced) * delta_reducHist     # max ir_length_s
    time_step = 1/sampling_rate_dirac
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
    print(f"Sampling rate dirac: {sampling_rate_dirac}\n" +
          f"Delta_redHist: {delta_reducHist}\nFactor delta_t: {factor_delta}")

    dirac_times = np.arange(0, time_stop, time_step)
    dirac_values = np.zeros_like(dirac_times)
    for time in diracNonZeros:
        ix = int(abs(time)/time_step)
        if dirac_values[ix] == 0:
            dirac_values[ix] = np.sign(time)

    # plot the dirac sequence in time and frequency domain
    pf.plot.time(
        (dirac_sig := pf.Signal(dirac_values, sampling_rate_dirac)),
        label="Dirac sequence")
    plt.xlim(0.0, 0.05) #for density checking
    plt.legend()
    plt.show()
    pf.plot.freq(dirac_sig, label="Dirac sequence")
    #plt.xlim(5, 24000)
    plt.legend()
    plt.show()

    print("Factor between histogram and dirac sequence sampling rate: " +
            f"{(factor_s := int(sampling_rate_dirac*delta_reducHist))}")
                # maybe check >1! and not float!

    # IEC 61260:1:2014 standard or in the future e.g. Raised Cosine Filter
    if divide_BW:
        filters = pf.dsp.filter.fractional_octave_bands(
            signal=None,
            num_fractions=1,
            frequency_range=(125, 16000),
            sampling_rate=sampling_rate_dirac,
        )
        dirac_filtered_sig = filters.process(dirac_sig)  # type: ignore
        centerFreq = pf.dsp.filter.fractional_octave_frequencies(
            frequency_range=(125, 16000))[1]
        pf.plot.freq(
            pf.Signal(dirac_filtered_sig.time, sampling_rate_dirac),
        )
        plt.show()

        dirac_weighted = np.zeros((dirac_filtered_sig.time.shape[0],
                                   dirac_filtered_sig.time.shape[2]))
        bw_size = [centerFreq[i]*np.sqrt(2) - centerFreq[i]/np.sqrt(2) for i
                        in range(len(centerFreq))]

        for filter_ix in range(dirac_weighted.shape[0]):
            print(f"Filter number: {filter_ix}")
            for sample_i in range(dirac_weighted.shape[1]):
                low = int(sample_i / factor_s) * factor_s
                high = int(sample_i / factor_s) * factor_s + factor_s - 1  # -1 right?
                div = sum(dirac_filtered_sig.time[filter_ix, 0, low:high] ** 2)

                dirac_weighted[filter_ix, sample_i] = (
                    dirac_filtered_sig.time[filter_ix, 0, sample_i]
                    * np.sqrt(hist_reduced[int(sample_i / factor_s)] / div)
                    * np.sqrt(bw_size[filter_ix] / (sampling_rate_dirac / 2))
                )

        pf.plot.time(pf.Signal(dirac_weighted, sampling_rate_dirac),
                     dB=True, label="Dirac sequence weighted filters",
                     log_prefix=20)
        plt.show()
        dirac_weighted = np.sum(dirac_weighted, axis=0) # sum over spectrum

    else:
        dirac_weighted = np.zeros_like(dirac_sig.time[0])
        ###  MISSING CODE ###
        ###  reduced copy from above

    pf.plot.time(
        (hist_weighted_sig := pf.Signal(dirac_weighted, sampling_rate_dirac)),
        dB=True,
        label="Dirac sequence weighted",
        log_prefix=20,
    )
    pf.plot.time(
        hist_reduced_sig,
        dB=True,
        log_prefix=10,
        label=f"Histogram reduced of {room_volume} m^3 room"
        + f"\nReduction of resolution by {factor_delta}x",
    )
    plt.ylim(-200, 0)
    #plt.xlim(0, 1) #for check
    plt.legend()
    #plt.savefig(fname=f"sim_data/f_dirac_weighted_{room_volume}m^3_{delta_redHist}s.svg")
    plt.show()
    txt_data = np.concatenate((dirac_times, dirac_weighted))
    txt_data = np.reshape(txt_data, (2, len(dirac_weighted)))
    np.savetxt("sim_data/f_ir_room_" + ".csv", txt_data, delimiter=",")

    pf.io.write_audio(
        hist_weighted_sig,
        f"sim_data/diracS_weighted_{room_volume}m^3_{delta_reducHist}s.wav",
        overwrite=True,
    )
    # wav-files of subtype PCM_16 are clipped to +/- 1. Normalize your audio with
    # pyfar.dsp.normalize to 1-LSB, with LSB being the least significant bit (e.g. 2**-15 for
    # 16 bit) or use non-clipping subtypes 'FLOAT', 'DOUBLE', or 'VORBIS'
    # (see pyfar.io.audio_subtypes)

    run_csv_comparison(
        "sim_data/f_hist_room_4_5_3_1_8k.csv",
        8000,
        "sim_data/f_ir_room_.csv",
        sampling_rate_dirac,
        res_reduction_data2=True,
        conv_to_energy_data2=True,
    )

    return hist_weighted_sig


# %% Run the functions
update_hist = False
flag_char = ""
delta_reduced_histogram = 0.001  # default 0.1 for the IR

X = 4
Y = 5
Z = 3
room_volume = X * Y * Z

ir_length_s = 1         # default 2 for update_hist
sampling_rate = 8000    # 1/delta_t below 48000
patch_size = 1
max_order_k = 140       # -200dB clean for >=180 at 2s, 140 at 1s for update_hist
print(f"X: {X}\nY: {Y}\nZ: {Z}\nPatch size: {patch_size}")
str_fileNamePath = (
    f"sim_data/f_hist_room_{X}_{Y}_{Z}_{patch_size}_{int(sampling_rate/1000)}k{flag_char}")

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
    print("Debug: returned")
    txt_data = np.concatenate(
        (
            np.arange(
                start=0, stop=hist_full.shape[3] / sampling_rate, step=1 / sampling_rate),
            hist_full[0, :, 0, :].reshape(-1),  # one freq bin
        ),
    )
    txt_data = np.reshape(txt_data, (hist_full.shape[1] + 1, hist_full.shape[3]))
    np.savetxt(str_fileNamePath + ".csv", txt_data, delimiter=",")
    delta = datetime.now() - start
    print(f"Time elapsed: {delta}")
else:
    txt_data = np.genfromtxt(str_fileNamePath + ".csv", delimiter=",")

hist_sum = np.sum(txt_data[1:, :], axis=0)

hist_sum_sig = pf.Signal(hist_sum, sampling_rate)
plt.figure()
pf.plot.time( hist_sum_sig, dB=True, log_prefix=10,
    label=f"Histogram of room with size {X}x{Y}x{Z} m\n"+
    f"Patch size of {patch_size}{flag_char}")
plt.ylim(-200, 0)
plt.xlabel("seconds")
plt.legend()
if update_hist:
    plt.savefig(fname= str_fileNamePath + ".svg")
plt.show()

check_what_pf_sig = run_ir_generation(
    hist_sum,
    room_volume,
    sampling_rate,
    keep_original_hist=False,
    delta_reducHist=delta_reduced_histogram,
)
