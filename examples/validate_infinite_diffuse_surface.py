# %%
import numpy as np
import sparrowpy as sp
import pyfar as pf
import matplotlib.pyplot as plt

# %%
# Setup a simulation for infinite plane

def calculate_ratio_old(
        width, depth, patch_size, source_pos, receiver_pos, speed_of_sound, sampling_rate):
    source = pf.Coordinates(*source_pos)
    receiver = pf.Coordinates(*receiver_pos)
    source_is = source.copy()
    source_is.z *= -1
    reflection_len =  (receiver - source_is).radius[0]
    max_histogram_length = reflection_len/speed_of_sound
    # print(max_histogram_length)
    max_histogram_length=1

    plane = sp.geometry.Polygon(
        [[-width/2, -depth/2, 0],
        [-width/2, depth/2, 0],
        [width/2, depth/2, 0],
        [width/2, depth/2, 0]],
        up_vector=np.array([1.,0.,0.]),
        normal=np.array([0.,0.,1.]))

    #simulation parameters
    radi = sp.radiosity.Radiosity(
        [plane], patch_size, 1, max_histogram_length,
        speed_of_sound=speed_of_sound, sampling_rate=sampling_rate)

    # run simulation
    source = sp.geometry.SoundSource(source_pos, [0, 1, 0], [0, 0, 1])
    radi.run(source)

    # gather energy at receiver
    receiver = sp.geometry.Receiver(receiver_pos, [0, 1, 0], [0, 0, 1])

    ir_slow = radi.energy_at_receiver(receiver, ignore_direct=True)
    I_diffuse = pf.Signal(ir_slow, sampling_rate=sampling_rate)
    I_specular = 1/(4*np.pi*reflection_len**2)
    return np.sum(I_diffuse.freq[:, :])/I_specular, I_specular, I_diffuse


def calculate_ratio_new(
        width, depth, patch_size, source_pos, receiver_pos, speed_of_sound, sampling_rate):
    source = pf.Coordinates(*source_pos)
    receiver = pf.Coordinates(*receiver_pos)
    source_is = source.copy()
    source_is.z *= -1
    reflection_len =  (receiver - source_is).radius[0]
    max_histogram_length = reflection_len/speed_of_sound
    # print(max_histogram_length)
    max_histogram_length=1

    plane = sp.geometry.Polygon(
        [[-width/2, -depth/2, 0],
        [-width/2, depth/2, 0],
        [width/2, depth/2, 0],
        [width/2, depth/2, 0]],
        up_vector=np.array([1.,0.,0.]),
        normal=np.array([0.,0.,1.]))

    #simulation parameters
    radi = sp.radiosity_fast.DRadiosityFast.from_polygon(
        [plane], patch_size)

    brdf_sources = pf.Coordinates(0, 0, 1, weights=1)
    brdf_receivers = pf.Coordinates(0, 0, 1, weights=1)
    brdf = sp.brdf.create_from_scattering(
        brdf_sources,
        brdf_receivers,
        pf.FrequencyData(1, [100]),
        pf.FrequencyData(0, [100]),
    )
    radi.set_wall_scattering(
        np.arange(1), brdf, brdf_sources, brdf_receivers)

    # set air absorption
    radi.set_air_attenuation(
        pf.FrequencyData(
            np.zeros_like(brdf.frequencies),
            brdf.frequencies))

    # set absorption coefficient
    radi.set_wall_absorption(
        np.arange(1),
        pf.FrequencyData(
            np.zeros_like(brdf.frequencies)+0,
            brdf.frequencies))

    # calculate from factors including directivity and absorption
    radi.bake_geometry(algorithm='order')

    # initialize source energy at each patch
    radi.init_source_energy(source_pos, algorithm='order')

    # gather energy at receiver
    radi.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        histogram_time_resolution=1/sampling_rate,
        histogram_length=max_histogram_length,
        algorithm='order', max_depth=2)
    ir_fast = radi.collect_receiver_energy(
        receiver_pos, speed_of_sound=speed_of_sound,
        histogram_time_resolution=1/sampling_rate,
        propagation_fx=True)
    I_diffuse = pf.Signal(ir_fast, sampling_rate=sampling_rate)

    I_specular = 1/(4*np.pi*reflection_len**2)
    return np.sum(I_diffuse.freq[:, :])/I_specular, I_specular, I_diffuse


# %%
# Test case one

width = 40
depth = 40
patch_size = 1
speed_of_sound = 346.18
sampling_rate = 1

ratio_1_new = []
ratio_1_old = []
ratio_2_new = []
ratio_2_old = []
ratio_3 = []
ratio_4 = []
dimensions = [1, 10, 100, 1000]
for w in dimensions:
    # case 1: source and receiver are at the same position
    expected_ratio = 2
    source_pos = [0, 0, 3]
    receiver_pos = [0, 0, 3]
    r, I_specular, I_diffuse = calculate_ratio_old(
        width, depth, patch_size, source_pos,
        receiver_pos, speed_of_sound, w)
    ratio_1_old.append(r)
    r, I_specular, I_diffuse = calculate_ratio_new(
        width, depth, patch_size, source_pos,
        receiver_pos, speed_of_sound, w)
    ratio_1_new.append(r)
    # case 2: source and receiver are on the same normal
    expected_ratio = 2
    source_pos = [0, 0, 10]
    receiver_pos = [0, 0, 3]
    r, I_specular, I_diffuse = calculate_ratio_old(
        width, depth, patch_size, source_pos,
        receiver_pos, speed_of_sound, w)
    ratio_2_old.append(r)
    r, I_specular, I_diffuse = calculate_ratio_new(
        width, depth, patch_size, source_pos,
        receiver_pos, speed_of_sound, w)
    ratio_2_new.append(r)


# %%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogx(dimensions, ratio_1_old, '--', label='case 1 old')
ax.semilogx(dimensions, ratio_1_new, label='case 1 new')
ax.semilogx(dimensions, ratio_2_old, '--', label='case 2 old')
ax.semilogx(dimensions, ratio_2_new, label='case 2 new')
ax.hlines(2, dimensions[0], dimensions[-1], 'r', label='expected')
ax.set_xlabel('sampling rate')
plt.legend()
plt.show()
fig.savefig('validate_infinite_diffuse_surface_vs_sampling_rate.png')



# %%

# %%
# Test case one

width = 10
depth = 10
patch_size = 1
speed_of_sound = 346.18
sampling_rate = 1

ratio_1_new = []
ratio_1_old = []
ratio_2_new = []
ratio_2_old = []
ratio_3 = []
ratio_4 = []
dimensions = [10, 20, 30, 40, 50]
for w in dimensions:
    # case 1: source and receiver are at the same position
    expected_ratio = 2
    source_pos = [0, 0, 3]
    receiver_pos = [0, 0, 3]
    r, I_specular, I_diffuse = calculate_ratio_old(
        w, w, patch_size, source_pos,
        receiver_pos, speed_of_sound, sampling_rate)
    ratio_1_old.append(r)
    r, I_specular, I_diffuse = calculate_ratio_new(
        w, w, patch_size, source_pos,
        receiver_pos, speed_of_sound, sampling_rate)
    ratio_1_new.append(r)
    # case 2: source and receiver are on the same normal
    expected_ratio = 2
    source_pos = [0, 0, 10]
    receiver_pos = [0, 0, 4]
    r, I_specular, I_diffuse = calculate_ratio_old(
        w, w, patch_size, source_pos,
        receiver_pos, speed_of_sound, sampling_rate)
    ratio_2_old.append(r)
    r, I_specular, I_diffuse = calculate_ratio_new(
        w, w, patch_size, source_pos,
        receiver_pos, speed_of_sound, sampling_rate)
    ratio_2_new.append(r)


# %%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dimensions, ratio_1_old, '--', label='case 1 old')
ax.plot(dimensions, ratio_1_new, label='case 1 new')
ax.plot(dimensions, ratio_2_old, '--', label='case 2 old')
ax.plot(dimensions, ratio_2_new, label='case 2 new')
ax.hlines(2, dimensions[0], dimensions[-1], 'r', label='expected')
plt.legend()
plt.show()
fig.savefig('validate_infinite_diffuse_surface_vs_plane_size.png')

# %%
