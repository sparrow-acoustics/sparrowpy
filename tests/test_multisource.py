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
    np.array([.25,.25,.25]),
    np.array([ np.array([.25,.25,.25]),
               np.array([.45,.45,.45]),
               np.array([.75,.75,.75]) ]),
    ])
def test_multi_receiver(basicscene, frequencies,
                        receivers, method="universal"):
    """Check validity of multiple receiver output."""
    algo= "order"

    src = np.array([.5,.5, .5])

    radi = run_basicscene(basicscene, src_pos=src, freqs=frequencies,
                          algorithm=algo, method=method)

    big_histo = radi.collect_receiver_energy(receiver_pos=receivers,
                                  speed_of_sound=basicscene["speed_of_sound"],
                                   histogram_time_resolution=1/basicscene["sampling_rate"],
                                   method=method,
                                   )

    # assert correct dimensions of output histogram
    if receivers.ndim==1:
        n_recs = 1
    else:
        n_recs = receivers.shape[0]

    assert big_histo.shape[0] == n_recs
    assert big_histo.shape[1] == radi._n_patches
    assert big_histo.shape[2] == frequencies.shape[0]

    # assert big histogram entries same as histograms for individual receiver
    if n_recs > 1:
        for i in range(n_recs):
            small_histo = radi.collect_receiver_energy(
                                    receiver_pos=receivers[i],
                                    speed_of_sound=basicscene["speed_of_sound"],
                                    histogram_time_resolution=1/basicscene["sampling_rate"],
                                    method=method,
                                    )

            npt.assert_array_almost_equal(big_histo[i], small_histo[0])



@pytest.mark.parametrize('src', [
    [2.,1.5,1.5],
    ])
@pytest.mark.parametrize('rec', [
    [.5,1,2.5],
    [1,2.5,1.5],
    ])
@pytest.mark.parametrize('order', [
    10,20,
    ])
@pytest.mark.parametrize('ps', [
    .5,1.5,
    ])
def test_reciprocity_shoebox(src,rec,order,ps, method="universal"):
    """Test if radiosity results are reciprocal in shoebox room."""
    X = 3
    Y = 3
    Z = 3
    patch_size = ps
    ir_length_s = .5
    sampling_rate = 200
    max_order_k = order
    speed_of_sound = 343
    irs_new = []
    frequencies = np.array([1000])
    absorption = 0.
    walls = sp.testing.shoebox_room_stub(X, Y, Z)
    algo= "order"

    sc_src = pf.Coordinates(0, 0, 1)
    sc_rec = pf.Coordinates(0, 0, 1)

    for i in range(2):
        if i == 0:
            src_ = np.array(src)
            rec_ = np.array(rec)
        elif i == 1:
            src_ = np.array(rec)
            rec_ = np.array(src)


        ## initialize radiosity class
        radi = sp.radiosity_fast.DirectionalRadiosityFast.from_polygon(walls, patch_size)

        source_brdf = pf.Coordinates(0, 0, 1, weights=1)
        receivers_brdf = pf.Coordinates(0, 0, 1, weights=1)
        brdf = sp.brdf.create_from_scattering(
            source_brdf,
            receivers_brdf,
            pf.FrequencyData(
                np.ones((frequencies.size)), frequencies))

        # set directional scattering data
        radi.set_wall_scattering(
            np.arange(len(walls)), brdf, sc_src, sc_rec)

        # set air absorption
        radi.set_air_attenuation(
            pf.FrequencyData(
                np.zeros_like(frequencies),
                frequencies))

        # set absorption coefficient
        radi.set_wall_absorption(
            np.arange(len(walls)),
            pf.FrequencyData(
                np.zeros_like(frequencies)+absorption,
                frequencies))

        # run simulation
        radi.bake_geometry(ff_method=method,algorithm=algo)

        radi.init_source_energy(src_,ff_method=method,algorithm=algo)

        radi.calculate_energy_exchange(
                            speed_of_sound=speed_of_sound,
                            histogram_time_resolution=1/sampling_rate,
                            histogram_length=ir_length_s, algorithm=algo,
                            max_depth=max_order_k,
                            )

        ir = np.sum(
            radi.collect_receiver_energy(receiver_pos=rec_,
                                        speed_of_sound=speed_of_sound,
                                        histogram_time_resolution=1/sampling_rate,
                                        method=method, propagation_fx=True,
                                        ),
                    axis=1)[0][0]

        # test energy at receiver
        irs_new.append(ir)

    irs_new = np.array(irs_new)

    npt.assert_array_almost_equal(np.sum(irs_new[1]), np.sum(irs_new[0]))
    npt.assert_array_almost_equal(irs_new[1], irs_new[0])


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
            e_s,_ = sp.radiosity_fast.source_energy._init_energy_universal(
                                                    source_position=src_.position,
                                                    patches_center=np.array([wall[0].center]),
                                                    patches_points=np.array([wall[0].pts]),
                                                    air_attenuation=air_att,
                                                    n_bins=1,
                                                    )

            e_r = sp.radiosity_fast.receiver_energy._universal(
                                                    receiver_pos=rec_.position,patches_points=np.array([wall[0].pts]),
                                                    )


        elif method == "kang":
            e_s,_ = sp.radiosity_fast.source_energy._init_energy_kang(
                                                        source_position=src_.position,
                                                        patches_center=np.array([wall[0].center]),
                                                        patches_normal=np.array([wall[0].normal]),
                                                        air_attenuation=air_att,
                                                        patches_size=np.array([wall[0].size]),
                                                        n_bins=1,
                                                        )

            e_r = sp.radiosity_fast.receiver_energy._kang(
                                                        patch_receiver_distance=np.array([(rec_.position-wall[0].center)]),
                                                        patches_normal=np.array([wall[0].normal]),
                                                        )

        e = e_s*e_r

        energy.append(e)

    npt.assert_array_almost_equal(energy[0], energy[1])


def run_basicscene(scene, src_pos, freqs, algorithm, method):
    patch_size = scene["patch_size"]
    ir_length_s = scene["ir_length_s"]
    sampling_rate = scene["sampling_rate"]
    max_order_k = scene["max_order_k"]
    speed_of_sound = scene["speed_of_sound"]
    absorption = scene["absorption"]
    walls = scene["walls"]

    sc_src = pf.Coordinates(0, 0, 1)
    sc_rec = pf.Coordinates(0, 0, 1)

    ## initialize radiosity class
    radi = sp.radiosity_fast.DirectionalRadiosityFast.from_polygon(walls, patch_size)

    data_scattering = pf.FrequencyData(
        np.ones((sc_src.csize,sc_rec.csize,freqs.size)), freqs)

    # set directional scattering data
    radi.set_wall_scattering(
        np.arange(len(walls)), data_scattering, sc_src, sc_rec)

    # set air absorption
    radi.set_air_attenuation(
        pf.FrequencyData(
            np.zeros_like(data_scattering.frequencies),
            data_scattering.frequencies))

    # set absorption coefficient
    radi.set_wall_absorption(
        np.arange(len(walls)),
        pf.FrequencyData(
            np.zeros_like(data_scattering.frequencies)+absorption,
            data_scattering.frequencies))

    # run simulation
    radi.bake_geometry(ff_method=method,algorithm=algorithm)

    radi.init_source_energy(src_pos,ff_method=method,algorithm=algorithm)

    radi.calculate_energy_exchange(
            speed_of_sound=speed_of_sound,
            histogram_time_resolution=1/sampling_rate,
            histogram_length=ir_length_s, algorithm=algorithm,
            max_depth=max_order_k )

    return radi
