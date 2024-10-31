"""Test the universal form factor module."""

import pytest
import sparapy.geometry as geo
import numpy as np
import numpy.testing as npt
import sparapy.radiosity_fast.universal_ff.univ_form_factor as form_factor
import sparapy.testing.exact_ff_solutions as exact_solutions
from sparapy.sound_object import SoundSource, Receiver
from sparapy.radiosity import Patches
import time
import sparapy as sp
import pyfar as pf
from sparapy.radiosity_fast import form_factor as FFac


@pytest.mark.parametrize("width", [0.5, 1.0, 1.5, 3.])
@pytest.mark.parametrize("height", [0.5, 1.0, 1.5, 3.])
@pytest.mark.parametrize("distance", [0.5, 1.0, 1.5, 3.])
def test_parallel_facing_patches(width, height, distance):
    """Test universal form factor for equal facing parallel patches."""
    exact = exact_solutions.parallel_patches(width, height, distance)

    patch_1 = geo.Polygon(
        points=[[0, 0, 0], [width, 0, 0], [width, 0, height], [0, 0, height]],
        normal=[0, 1, 0],
        up_vector=[1, 0, 0],
    )

    patch_2 = geo.Polygon(
        points=[
            [0, distance, 0],
            [width, distance, 0],
            [width, distance, height],
            [0, distance, height],
        ],
        normal=[0, -1, 0],
        up_vector=[1, 0, 0],
    )

    kang = FFac.kang(patches_center=np.array([patch_1.center, patch_2.center]),
                     patches_normal=np.array([patch_1.normal, patch_2.normal]),
                     patches_size=np.array([[width,height],[width, height]]),
                     visible_patches=np.array([[0,1]]))

    univ = FFac.universal(patches_points=np.array([patch_1.pts, patch_2.pts]),
                          patches_normals=np.array(
                              [patch_1.normal, patch_2.normal]),
                          patches_areas=np.array([patch_1.area, patch_2.area]),
                          visible_patches=np.array([[0,1]]))

    assert 100 * abs(univ[0,1] - exact) / exact < 20

    assert abs(univ[0,1] - exact) < abs(kang[0,1] - exact)


@pytest.mark.parametrize("width", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("height", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("length", [1.0, 2.0, 3.0])
def test_perpendicular_coincidentline_patches(width, height, length):
    """Test universal form factor for perpendicular patches sharing a side."""
    exact = exact_solutions.perpendicular_patch_coincidentline(
        width, height, length
    )

    patch_1 = geo.Polygon(
        points=[[0, 0, 0], [width, 0, 0], [width, length, 0], [0, length, 0]],
        normal=[0, 0, 1],
        up_vector=[1, 0, 0],
    )

    patch_2 = geo.Polygon(
        points=[
            [0, 0, 0],
            [0, length, 0],
            [0, length, height],
            [0, 0, height],
        ],
        normal=[1, 0, 0],
        up_vector=[1, 0, 0],
    )

    univ = FFac.universal(patches_points=np.array([patch_1.pts, patch_2.pts]),
                          patches_normals=np.array(
                              [patch_1.normal, patch_2.normal]),
                          patches_areas=np.array([patch_1.area, patch_2.area]),
                          visible_patches=np.array([[0,1]]))

    assert 100 * abs(univ[0,1] - exact) / exact < 5


@pytest.mark.parametrize("width1", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("width2", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("length1", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("length2", [1.0, 2.0, 3.0])
def test_perpendicular_coincidentpoint_patches(
    width1, width2, length1, length2
):
    """Test form factor for perpendicular patches w/ common vertex."""
    exact = exact_solutions.perpendicular_patch_coincidentpoint(
        width1, length2, width2, length1
    )

    patch_1 = geo.Polygon(
        points=[
            [0, 0, 0],
            [0, length1, 0],
            [0, length1, width1],
            [0, 0, width1],
        ],
        normal=[1, 0, 0],
        up_vector=[0, 1, 0],
    )

    patch_2 = geo.Polygon(
        points=[
            [0, length1, 0],
            [width2, length1, 0],
            [width2, length1 + length2, 0],
            [0, length1 + length2, 0],
        ],
        normal=[0, 0, 1],
        up_vector=[1, 0, 0],
    )

    univ = FFac.universal(patches_points=np.array([patch_1.pts, patch_2.pts]),
                          patches_normals=np.array(
                              [patch_1.normal, patch_2.normal]),
                          patches_areas=np.array([patch_1.area, patch_2.area]),
                          visible_patches=np.array([[0,1]]))

    assert 100 * abs(univ[0,1] - exact) / exact < 5


@pytest.mark.parametrize("side", [0.1, 0.2, 0.5, 1, 2])
@pytest.mark.parametrize(
    "source",
    [
        SoundSource(
            position=np.array([1, 7, 1]),
            view=np.array([0, -1, 0]),
            up=np.array([0, 0, 1]),
        ),
    ],
)
@pytest.mark.parametrize(
    "receiver",
    [
        Receiver(
            position=np.array([1, 1, 1]),
            view=np.array([0, -1, 0]),
            up=np.array([0, 0, 1]),
        ),
    ],
)
@pytest.mark.parametrize("patchsize", [0.1])
def test_point_surface_interactions(side, source, receiver, patchsize):
    """Test source-to-patch and patch-to-receiver factor calculation."""
    sr = 1000
    c = 343

    absor_factor = 0.1

    patch_pos = geo.Polygon(
        points=[[0, 0, 0], [side, 0, 0], [side, 0, side], [0, 0, side]],
        normal=[0, 1, 0],
        up_vector=[1, 0, 0],
    )

    patch = Patches(
        polygon=patch_pos,
        max_size=patchsize * side,
        other_wall_ids=[],
        wall_id=[0],
        absorption=absor_factor,
    )

    patch.init_energy_exchange(
        0, 0.1, source, sampling_rate=sr, speed_of_sound=c
    )

    patch = source_cast(src=source, rpatch=patch, absor=absor_factor)

    receiver_cast(receiver, patch, absor_factor, sr, c)


def source_cast(src, rpatch, absor):
    """Cast and test source-to-patch factor calculation."""
    t0 = time.time()
    nuss = form_factor.pt_solution(point=src.position, patch_points=rpatch.pts)
    tf_nusselt = time.time() - t0

    true = sum(rpatch.E_matrix[rpatch.E_matrix != 0])

    rel_error_nuss = abs(true - nuss * (1 - absor)) / true * 100

    assert rel_error_nuss < 1.0

    print("nusselt approach error: " + str(rel_error_nuss) + "%")
    print(
        "nusselt approach runtime: "
        + str(tf_nusselt * 1000)
        + "ms \n #################################"
    )
    return rpatch


def receiver_cast(rcv, patch, radi, sr, c):
    """Cast and test patch-to-receiver factor calculation."""
    true_rec_energy = np.sum(
        patch.energy_at_receiver(
            receiver=rcv,
            max_order=0,
            ir_length_s=0.1,
            speed_of_sound=c,
            sampling_rate=sr,
        )
    )

    patch_energy = np.sum(patch.E_matrix)

    nuss = form_factor.pt_solution(
            point=rcv.position, patch_points=patch.pts, mode="receiver"
            ) * patch_energy

    rel_error_nuss = abs(true_rec_energy - nuss) / true_rec_energy * 100

    assert rel_error_nuss < 1.0


@pytest.mark.parametrize(
    "src",
    [
        [[2.0, 1.0, 1.0], [0, 1, 0], [0, 0, 1]],
        [[0.5, 0.5, 0.5], [-1, 1, 0], [0, 0, 1]],
        [[0.5, 2, 2.5], [1, 0, 0], [0, 0, 1]],
    ],
)
@pytest.mark.parametrize(
    "rec",
    [
        [[1.0, 1.0, 0.5], [-1, 1, 0], [0, 0, 1]],
    ],
)
def test_fast_ff_method_comparison(src, rec):
    """Test if the radiosity results differ significanly between ff methods."""
    X = 3
    Y = 3
    Z = 3
    patch_size = 1
    ir_length_s = 1
    sampling_rate = 100
    max_order_k = 10
    speed_of_sound = 343
    irs_new = []
    frequencies = np.array([1000])
    absorption = 0.0
    walls = sp.testing.shoebox_room_stub(X, Y, Z)
    algo = "order"

    sc_src = pf.Coordinates(0, 0, 1)
    sc_rec = pf.Coordinates(0, 0, 1)

    src_ = sp.geometry.SoundSource(src[0], src[1], src[2])
    rec_ = sp.geometry.Receiver(rec[0], rec[1], rec[2])

    for method in ["kang", "universal"]:
        ## initialize radiosity class
        radi = sp.radiosity_fast.DRadiosityFast.from_polygon(walls, patch_size)

        data_scattering = pf.FrequencyData(
            np.ones((sc_src.csize, sc_rec.csize, frequencies.size)),
            frequencies,
        )

        # set directional scattering data
        radi.set_wall_scattering(
            np.arange(len(walls)), data_scattering, sc_src, sc_rec
        )

        # set air absorption
        radi.set_air_attenuation(
            pf.FrequencyData(
                np.zeros_like(data_scattering.frequencies),
                data_scattering.frequencies,
            )
        )

        # set absorption coefficient
        radi.set_wall_absorption(
            np.arange(len(walls)),
            pf.FrequencyData(
                np.zeros_like(data_scattering.frequencies) + absorption,
                data_scattering.frequencies,
            ),
        )

        # run simulation
        radi.bake_geometry(ff_method=method, algorithm=algo)

        radi.init_source_energy(
            src_.position, ff_method=method, algorithm=algo
        )

        ir = radi.calculate_energy_exchange_receiver(
            receiver_pos=rec_.position,
            speed_of_sound=speed_of_sound,
            histogram_time_resolution=1 / sampling_rate,
            histogram_length=ir_length_s,
            ff_method=method,
            algorithm=algo,
            max_depth=max_order_k,
        )

        # test energy at receiver
        irs_new.append(ir)

    irs_new = np.array(irs_new)

    npt.assert_allclose(irs_new[1][0], irs_new[0][0],rtol=.1)
