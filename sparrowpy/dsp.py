import sparrowpy as sp
import pyfar as pf #type: ignore
import numpy as np
import matplotlib.pyplot as plt
# ruff: noqa: D100


def run_energy_simulation_room(
    dimensions,
    patch_size,
    ir_length_s,
    sampling_rate,
    max_order_k,
    source,
    receiver,
    speed_of_sound=343,
) -> np.ndarray:
    """
    Calculate the histogram for a room with dimensions.

    Parameters
    ----------
    dimensions : ndarray:float[X, Y, Z]
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
    speed_of_sound : float, optional
        Speed of sound.

    Returns
    -------
    histogram : NDArray[n_receivers, n_patches, n_(freq)bins, E_matrix_total '-1'shape]
        Histogram of the room.
    """
    walls = sp.testing.shoebox_room_stub(dimensions[0], dimensions[1], dimensions[2])
    radiosity_model = sp.DRadiosityFast.from_polygon(walls, patch_size)

    # maybe problems with the brdf directions for different norm vectors
    brdf = sp.brdf.create_from_scattering(
        source_directions=pf.Coordinates(0, 0, 1, weights=1),
        receiver_directions=pf.Coordinates(0, 0, 1, weights=1),
        scattering_coefficient=pf.FrequencyData(1, [1000]),
        absorption_coefficient=pf.FrequencyData(0, [1000]),
    )
    radiosity_model.set_wall_scattering(
        wall_indexes=np.arange(len(walls)).tolist(),
        scattering=brdf,
        sources=pf.Coordinates(0, 0, 1, weights=1),
        receivers=pf.Coordinates(0, 0, 1, weights=1),
    )
    radiosity_model.set_air_attenuation(
        pf.FrequencyData([0.2], [1000]),
    )
    radiosity_model.set_wall_absorption(
        np.arange(len(walls)).tolist(),
        pf.FrequencyData(np.zeros_like(brdf.frequencies), brdf.frequencies),
    )
    radiosity_model.bake_geometry()
    radiosity_model.init_source_energy(source)
    radiosity_model.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        histogram_time_resolution=1 / sampling_rate,
        histogram_length=ir_length_s,
        max_depth=max_order_k,
    )
    histogram = radiosity_model.collect_receiver_energy(
        receiver_pos=receiver,
        speed_of_sound=speed_of_sound,
        histogram_time_resolution=1 / sampling_rate,
        propagation_fx=True,
    )

    return histogram



def histogram_resolution_reduction(
    histogram,
    sampling_rate,
    sampling_rate_new,
    show_plots=True,
    ) -> pf.Signal:
    """
    Reduce the resolution of a histogram to certain sampling rate.
    Division of sampling rates should be a zero remainder.

    Parameters
    ----------
    histogram : NDArray[n_receivers, n_patches, n_(freq)bins, E_matrix_total '-1'shape]
        Histogram of the radiosity model.
    sampling_rate : float
        Sampling rate of the original histogram.
    sampling_rate_new : float
        Sampling rate of the new histogram.
    show_plots : bool, optional
        Show data plots.
    """
    if sampling_rate_new > sampling_rate:
         raise ValueError("New sampling rate is higher than the original one!")

    factor_delta = sampling_rate / sampling_rate_new
    print(f"Reduction in histogram resolution: {factor_delta}")
    hist_reduced = []
    index_values = range(0, len(histogram), int(factor_delta))
    for ix in index_values:
        hist_reduced.append(sum(histogram[ix : ix + int(factor_delta)]))
    hist_reduced_sig = pf.Signal(hist_reduced, sampling_rate_new)

    if show_plots:
        pf.plot.time(
            hist_reduced_sig,
            dB=True,
            log_prefix=10,
            label=f"Histogram resolution reduction by {factor_delta}x",
        )
        plt.scatter(
            np.arange(0, len(hist_reduced) / sampling_rate_new, 1 / sampling_rate_new),
            np.multiply(10, np.log10(np.asarray(hist_reduced))),
            label="Separate values of reduced histogram",
        )
        plt.ylim(-200, 0)
        plt.legend()
        plt.show()

    return hist_reduced_sig



def generate_dirac_sequence_raven(
    room_volume,
    speed_of_sound,
    ir_length_s_stop,
    sampling_rate_dirac=48000,
    show_plots=True,
    ) -> pf.Signal:
    """Dirac generation for impulse response of raven implementation.

    Parameters
    ----------
    room_volume : float
        Value for the calculation of the mean reflection ratio in a room for a given time.
    speed_of_sound : float
        Speed of sound in the room.
    ir_length_s_stop : float
        Length of data to be generated in the histogram.
    sampling_rate_dirac : int, optional
        Temporal resolution of the sequence
        Limitation of freq_max, by default 48000 (48kHz)
    show_plots : bool, optional
        Show data plots.

    Returns
    -------
    pf.Signal
        Signal of the generated dirac impulse sequence.
    """
    if sampling_rate_dirac < 48000:
        raise ValueError("Sampling rate too low, incaccurate results.")

    diracNonZeros = []  # list of time data of each dirac impulse
    rng = np.random.default_rng()
    µ_t = 4 * np.pi * pow(speed_of_sound, 3) / room_volume

    time_start = 1/sampling_rate_dirac # division by zero prevention
    time_step = 1/sampling_rate_dirac
    for time in np.arange(time_start, ir_length_s_stop, time_step):
        time_of_itr = time
        while (delta :=\
            (1/(min(µ_t * pow(time, 2), 10000)) * np.log(1/rng.uniform(1e-10, 1)))) <\
                time_of_itr+time_step-time:
            time += delta
            diracNonZeros.append(rng.choice([-time, time], p=[0.5,0.5]))
            # +1 or -1 dirac is chosen randomly with equal probability

    dirac_values = np.zeros(ir_length_s_stop*time_step)
    for time in diracNonZeros:
        ix = int(abs(time) / time_step)
        if dirac_values[ix] == 0:
            dirac_values[ix] = np.sign(time)
    dirac_values_sig = pf.Signal(dirac_values, sampling_rate_dirac)

    if show_plots:
        pf.plot.time(dirac_values_sig, label="Dirac sequence")
        plt.xlim(0.0, 0.05)  # for density checking
        plt.legend()
        plt.show()
        pf.plot.freq(dirac_values_sig, label="Dirac sequence")
        plt.legend()
        plt.show()

    return dirac_values_sig



def dirac_band_filter(
    dirac_sig,
    frequency_range=[125, 16000],
    show_plots=True,
    ) -> tuple[pf.Signal, list[float]]:
    """Filtering of the dirac sequence.

    Parameters
    ----------
    dirac_sig : pf.Signal
        Dirac sequence signal to be filtered.
    frequency_range : list, optional
        Frequency range for filters, by default [125, 16000]
    show_plots : bool, optional
        Show the frequency band filtered diracs, by default True

    Returns
    -------
    tuple[pf.Signal, list[float]]
        Filtered dirac sequences and corresponding center frequencies.
    """
    if dirac_sig.sampling_rate < 48000:
        raise ValueError("Sampling rate too low, bad filtered results.")

    filters = pf.dsp.filter.fractional_octave_bands(
        signal=None,
        num_fractions=1,
        frequency_range=frequency_range,
        sampling_rate=dirac_sig.sampling_rate,
    )
    dirac_filtered_sig = filters.process(dirac_sig)  # type: ignore
    centerFreq = pf.dsp.filter.fractional_octave_frequencies(
        frequency_range=(125, 16000))[1]
    if show_plots:
        pf.plot.freq(dirac_filtered_sig, legend="Filtered dirac sequence")
        plt.legend()
        plt.show()

    return dirac_filtered_sig, centerFreq


def dirac_weighted_with_filters(
    dirac_filt_sig,
    centerFreq,
    histogram,
    show_plots=True,
)   -> tuple[pf.Signal, pf.Signal]:
    """Filtered dirac sequence weighted with histogram energy.

    Parameters
    ----------
    dirac_filt_sig : pf.Signal
        Filtered dirac sequence signal.
    centerFreq : array : float
        Center frequencies of the filters.
    histogram : pf.Signal
        Histogram of the radiosity model.
    show_plots : bool, optional
        Show the weighted dirac sequence, by default True

    Returns
    -------
    tuple[pf.Signal, pf.Signal]
        Filtered and weighted diracs -> IR Amplitude [separate bands, combined].
    """
    factor_s = int(dirac_filt_sig.sampling_rate/histogram.sampling_rate)
    print(f"Filtered diracs sampling rate ({dirac_filt_sig.sampling_rate})"+
        f"\ndivided by the energy histogram resolution to: {factor_s}")

    dirac_weighted = np.zeros((dirac_filt_sig.time.shape[0],
                                   dirac_filt_sig.time.shape[2]))
    bw_size = [centerFreq[i]*np.sqrt(2) - centerFreq[i]/np.sqrt(2) for i
                        in range(len(centerFreq))]

    for filt_ix in range(dirac_weighted.shape[0]):
        print(f"Filter number: {filt_ix}")
        for sample_i in range(dirac_weighted.shape[1]):
            low = int(sample_i / factor_s) * factor_s
            high = int(sample_i / factor_s) * factor_s + factor_s - 1
            div = sum(dirac_filt_sig.time[filt_ix, 0, low:high] ** 2)

            dirac_weighted[filt_ix, sample_i] = (
                dirac_filt_sig.time[filt_ix, 0, sample_i]
                * np.sqrt(histogram[int(sample_i / factor_s)] / div)
                * np.sqrt(bw_size[filt_ix] / (dirac_filt_sig.sampling_rate / 2))
            )
    ir_bands_sig = pf.Signal(dirac_weighted,
                             dirac_filt_sig.sampling_rate)
    ir_wide_sig = pf.Signal(np.sum(dirac_weighted, axis=0),
                            dirac_filt_sig.sampling_rate)  # sum over spectrum

    if show_plots:
        pf.plot.time(ir_bands_sig, dB=True, log_prefix=20,
                     label="IR bands of dirac sequence weighted")
        plt.show()
        pf.plot.time(ir_wide_sig, dB=True, log_prefix=20,
                     label="IR full spectrum")
        plt.show()

    return ir_bands_sig, ir_wide_sig



def dirac_weighted_no_filters(
    dirac_sig,
    histogram,
    show_plots=True,
)   -> pf.Signal:
    """Original dirac sequence weighted with histogram energy.

    Parameters
    ----------
    dirac_sig : pf.Signal
        Unalterd dirac sequence signal.
    histogram : pf.Signal
        Histogram of the radiosity model.
    show_plots : bool, optional
        Show the weighted dirac sequence, by default True

    Returns
    -------
    pf.Signal
        Weighted diracs -> IR Amplitude.
    """
    ####check dims of dirac_sig for access to time
    factor_s = int(dirac_sig.sampling_rate/histogram.sampling_rate)
    print(f"Filtered diracs sampling rate ({dirac_sig.sampling_rate})"+
        f"\ndivided by the energy histogram resolution to: {factor_s}")

    dirac_weighted = np.zeros_like(dirac_sig.time[0])
    for sample_i in range(len(dirac_weighted)):
        low = int(sample_i / factor_s) * factor_s
        high = int(sample_i / factor_s) * factor_s + factor_s - 1
        div = sum(dirac_sig.time[0, low:high] ** 2)

        dirac_weighted[sample_i] = (dirac_sig.time[0, sample_i]
            * np.sqrt(histogram[int(sample_i / factor_s)] / div))
    ir_sig = pf.Signal(dirac_weighted, dirac_sig.sampling_rate)

    if show_plots:
        pf.plot.time(ir_sig, dB=True, log_prefix=20, label="IR of dirac sequence weighted")
        plt.show()

    return ir_sig
