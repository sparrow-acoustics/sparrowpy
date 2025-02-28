import sparrowpy as sp
import pyfar as pf
import numpy as np

def generate_dirac_impulses(room_volume, speed_of_sound, ir_length_s_stop, sampling_rate_dirac=48000) -> pf.Signal:
    rng = np.random.default_rng()
    diracNonZeros = []
    µ_t = 4 * np.pi * pow(speed_of_sound, 3) / room_volume ##div by 1000 for testing

    time_start = 1/max(sampling_rate_dirac,1000)      # max ~0.3m sound travel time
    time_step = 1/sampling_rate_dirac
    for time in np.arange(time_start, ir_length_s_stop, time_step):
        time_for_itr = time
        while (delta :=\
            (1/(min(µ_t * pow(time, 2), 10000)) * np.log(1/rng.uniform(1e-10, 1)))) <\
                time_for_itr+time_step-time:
            time += delta
            diracNonZeros.append(rng.choice([-time, time], p=[0.5,0.5]))
            
    return pf.Signal([1], 1)
