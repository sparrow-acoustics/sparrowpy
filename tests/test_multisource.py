"""Test multiple source capabilities."""
import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
import sparrowpy as sp

create_reference_files = False

@pytest.mark.parametrize('frequencies', [
    np.array([1000]),
    np.array([500,1000,2000]),
    ])
@pytest.mark.parametrize('receivers', [
    pf.Coordinates(
        [.25,.45,.75],
        [.45,.45,.45],
        [.75,.75,.75],
    ),
])
def test_multi_receiver(basicscene, frequencies,
                        receivers):
    """Check validity of multiple receiver output."""

    src = pf.Coordinates(.5, .5, .5)

    radi = run_basicscene(
        basicscene, src_pos=src, freqs=frequencies,
    )

    big_histo = radi.collect_energy_receiver_mono(
        receivers=receivers,
    )

    # assert correct dimensions of output histogram
    n_recs = receivers.cshape[0]

    assert big_histo.cshape[0] == n_recs
    assert big_histo.cshape[1] == frequencies.shape[0]

    # assert big histogram entries same as histograms for individual receiver
    for i in range(n_recs):
        small_histo = radi.collect_energy_receiver_mono(
            receivers=receivers[i],
            )

        npt.assert_array_almost_equal(
            big_histo[i].time, small_histo[0].time)


@pytest.mark.parametrize('src', [
    pf.Coordinates(2, 1.5, 1.5),
    ])
@pytest.mark.parametrize('rec', [
    pf.Coordinates(.5, 1, 2.5),
    pf.Coordinates(1, 2.5, 1.5),
    ])
@pytest.mark.parametrize('order', [
    10, 20,
    ])
@pytest.mark.parametrize('ps', [
    1, 1.5,
    ])
def test_reciprocity_shoebox(src,rec,order,ps):
    """Test if radiosity results are reciprocal in shoebox room."""
    X = 3
    Y = 3
    Z = 3
    patch_size = ps
    ir_length_s = .5
    sampling_rate = 200
    max_order_k = order
    speed_of_sound = 343
    etcs_new = []
    frequencies = np.array([1000])
    absorption = 0.
    walls = sp.testing.shoebox_room_stub(X, Y, Z)

    sc_src = pf.Coordinates(0, 0, 1)
    sc_rec = pf.Coordinates(0, 0, 1)

    for i in range(2):
        if i == 0:
            src_ = src
            rec_ = rec
        elif i == 1:
            src_ = rec
            rec_ = src


        ## initialize radiosity class
        radi = sp.DirectionalRadiosityFast.from_polygon(
            walls, patch_size)

        source_brdf = pf.Coordinates(0, 0, 1, weights=1)
        receivers_brdf = pf.Coordinates(0, 0, 1, weights=1)
        brdf = sp.brdf.create_from_scattering(
            source_brdf,
            receivers_brdf,
            pf.FrequencyData(
                np.ones((frequencies.size)), frequencies),
            pf.FrequencyData(
                np.zeros((frequencies.size))+absorption, frequencies),
            )

        # set directional scattering data
        radi.set_wall_brdf(
            np.arange(len(walls)), brdf, sc_src, sc_rec)

        # set air absorption
        radi.set_air_attenuation(
            pf.FrequencyData(
                np.zeros_like(frequencies),
                frequencies))


        # run simulation
        radi.bake_geometry()

        radi.init_source_energy(src_)

        radi.calculate_energy_exchange(
                            speed_of_sound=speed_of_sound,
                            etc_time_resolution=1/sampling_rate,
                            etc_duration=ir_length_s,
                            max_reflection_order=max_order_k,
                            )

        etc = radi.collect_energy_receiver_mono(rec_)

        # test energy at receiver
        etcs_new.append(etc.time[0])

    etcs_new = np.array(etcs_new)

    npt.assert_array_almost_equal(np.sum(etcs_new[1]), np.sum(etcs_new[0]))
    npt.assert_array_almost_equal(etcs_new[1], etcs_new[0])


@pytest.mark.parametrize('src', [
    [[2.,0,0], [-1, 0, 0], [0, 0, 1]],
    [[2.,2.,0], [-1, 0, 0], [0, 0, 1]],
    [[2.,0.,2.], [-1, 0, 0], [0, 0, 1]],
    ])
@pytest.mark.parametrize('rec', [
    [[1.,0.,0], [-1, 0, 0], [0, 0, 1]],
    [[2.,-2.,0], [-1, 0, 0], [0, 0, 1]],
    [[2.,0.,-2.], [-1, 0, 0], [0, 0, 1]],
    ])
def test_reciprocity_s2p_p2r(src,rec,method="universal"):
    """Check if radiosity implementation has source-receiver reciprocity."""
    wall = [sp.geometry.Polygon(
            [[0, -1, -1], [0, -1, 1],
            [0, 1, 1], [0, 1, -1]],
            [0, 0, 1], [1, 0, 0])]

    attenuation = pf.FrequencyData(
                np.zeros((1,1,1)),
                np.array([1000]))

    air_att = np.atleast_1d(attenuation.freq.squeeze())

    energy=[]

    for i in range(2):
        if i == 0:
            src_ = sp.geometry.SoundSource(src[0],src[1], src[2])
            rec_ = sp.geometry.Receiver(rec[0],rec[1], rec[2])
        elif i == 1:
            src_ = sp.geometry.SoundSource(rec[0],rec[1], rec[2])
            rec_ = sp.geometry.Receiver(src[0],src[1], src[2])

        if method == "universal":
            e_s,_ = sp.form_factor.universal._source2patch_energy_universal(
                                                    source_position=src_.position,
                                                    patches_center=np.array([wall[0].center]),
                                                    patches_points=np.array([wall[0].pts]),
                                                    source_visibility=np.ones((wall[0].center.shape[0]),dtype=bool),
                                                    air_attenuation=air_att,
                                                    n_bins=1,
                                                    )

            e_r = sp.form_factor.universal._patch2receiver_energy_universal(
                                                    receiver_pos=rec_.position,
                                                    patches_points=np.array([wall[0].pts]),
                                                    receiver_visibility=np.array([True]),
                                                    )


        elif method == "kang":
            e_s,_ = sp.form_factor.kang._source2patch_energy_kang(
                                                        source_position=src_.position,
                                                        patches_center=np.array([wall[0].center]),
                                                        patches_normal=np.array([wall[0].normal]),
                                                        air_attenuation=air_att,
                                                        patches_size=np.array([wall[0].size]),
                                                        n_bins=1,
                                                        )

            e_r = sp.form_factor.kang._patch2receiver_energy_kang(
                                                        patch_receiver_distance=np.array([(rec_.position-wall[0].center)]),
                                                        patches_normal=np.array([wall[0].normal]),
                                                        )

        e = e_s*e_r

        energy.append(e)

    npt.assert_array_almost_equal(energy[0], energy[1])


def run_basicscene(scene, src_pos, freqs):
    patch_size = scene["patch_size"]
    ir_length_s = scene["ir_length_s"]
    sampling_rate = scene["sampling_rate"]
    max_order_k = scene["max_order_k"]
    speed_of_sound = scene["speed_of_sound"]
    absorption = scene["absorption"]
    walls = scene["walls"]

    sc_src = pf.Coordinates(0, 0, 1, weights=1)
    sc_rec = pf.Coordinates(0, 0, 1, weights=1)

    ## initialize radiosity class
    radi = sp.DirectionalRadiosityFast.from_polygon(
        walls, patch_size)

    s = pf.FrequencyData(np.ones((freqs.size)), freqs)
    alpha = pf.FrequencyData(np.zeros((freqs.size))+absorption, freqs)
    brdf = sp.brdf.create_from_scattering(
        sc_src,
        sc_rec,
        s,
        alpha,
        )

    # set directional scattering data
    radi.set_wall_brdf(
        np.arange(len(walls)), brdf, sc_src, sc_rec)

    # set air absorption
    radi.set_air_attenuation(
        pf.FrequencyData(
            np.zeros_like(brdf.frequencies),
            brdf.frequencies))

    # run simulation
    radi.bake_geometry()

    radi.init_source_energy(src_pos)

    radi.calculate_energy_exchange(
            speed_of_sound=speed_of_sound,
            etc_time_resolution=1/sampling_rate,
            etc_duration=ir_length_s,
            max_reflection_order=max_order_k)

    return radi
