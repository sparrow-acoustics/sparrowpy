"""Energy delayed beyond the end of the histogram must be discarded.

Delays must never wrap energy around to the start of the histogram (as
``np.roll`` does) and must not raise an IndexError.
"""
import numpy as np
import numpy.testing as npt
import pyfar as pf

from sparrowpy.classes.RadiosityFast import (
    _accumulate_direct_sound_into_bins,
    _collect_receiver_energy,
    _energy_exchange_init_energy,
)
from sparrowpy.classes.RadiosityKang import _add_delay


def test_collect_receiver_energy_shifts_and_discards_tail():
    n_samples = 10
    speed_of_sound = 1.
    time_resolution = 1.
    E = np.zeros((2, 1, n_samples))
    E[:, 0, 0] = 1.
    E[:, 0, -1] = 2.
    distance = np.array([3., 2 * n_samples])
    out = _collect_receiver_energy(
        E, distance, speed_of_sound, time_resolution, np.zeros(1))

    expected = np.zeros_like(E)
    # patch 0: delayed by 3 samples, the last sample falls out
    expected[0, 0, 3] = 1.
    # patch 1: delayed beyond the histogram, nothing remains
    npt.assert_array_equal(out, expected)


def test_collect_receiver_energy_without_delay():
    E = np.arange(10, dtype=float).reshape(1, 1, 10)
    out = _collect_receiver_energy(E, np.array([0.]), 1., 1., np.zeros(1))
    npt.assert_array_equal(out, E)


def test_energy_exchange_init_energy_discards_late_energy():
    energy_0 = np.ones((2, 1, 1))
    distance_0 = np.array([3., 20.])
    E = _energy_exchange_init_energy(10, energy_0, distance_0, 1., 1.)
    expected = np.zeros((2, 1, 1, 10))
    expected[0, 0, 0, 3] = 1.
    npt.assert_array_equal(E, expected)


def test_accumulate_direct_sound_discards_late_energy(six_axis_unit):
    out = np.zeros((2, 6, 1, 10))
    rec_xyz = np.array([[1., 0, 0], [1., 0, 0]])
    _accumulate_direct_sound_into_bins(
        out, rec_xyz, np.zeros(3), six_axis_unit,
        np.ones((2, 1)), np.array([3, 20]))
    assert np.sum(out[0]) == 1.
    assert np.sum(out[0, :, :, 3]) == 1.
    # no energy is clipped onto the last sample
    npt.assert_array_equal(out[1], 0)


def test_add_delay_discards_tail():
    ir = np.arange(1, 6, dtype=float)
    npt.assert_array_equal(_add_delay(ir, 2), [0, 0, 1, 2, 3])
    npt.assert_array_equal(_add_delay(ir, 0), ir)
    npt.assert_array_equal(_add_delay(ir, 5), np.zeros(5))
    npt.assert_array_equal(_add_delay(ir, 7), np.zeros(5))


def test_add_delay_axis_0():
    ir = np.arange(1, 7, dtype=float).reshape(3, 2)
    expected = np.array([[0, 0], [1, 2], [3, 4]], dtype=float)
    npt.assert_array_equal(_add_delay(ir, 1, axis=0), expected)


def test_collect_energy_receiver_mono_direct_sound_out_of_range():
    """Direct sound arriving after the ETC ends is dropped, no error."""
    import sparrowpy as sp
    X, Y, Z = 2, 2, 2
    walls = sp.testing.shoebox_room_stub(X, Y, Z)
    radi = sp.DirectionalRadiosityFast.from_polygon(walls, 1)
    brdf_sources = pf.Coordinates(0, 0, 1, weights=1)
    brdf_receivers = pf.Coordinates(0, 0, 1, weights=1)
    brdf = sp.brdf.create_from_scattering(
        brdf_sources, brdf_receivers,
        pf.FrequencyData(1, [100]), pf.FrequencyData(0, [100]))
    radi.set_wall_brdf(
        np.arange(len(walls)), brdf, brdf_sources, brdf_receivers)
    radi.bake_geometry()
    source = pf.Coordinates(0.5, 0.5, 0.5)
    radi.init_source_energy(source)
    # ETC is shorter than the direct sound delay (~2.5 ms)
    radi.calculate_energy_exchange(
        speed_of_sound=343, etc_time_resolution=1e-3, etc_duration=2e-3,
        max_reflection_order=0)
    etc = radi.collect_energy_receiver_mono(
        pf.Coordinates(1.5, 1.5, 1.5), direct_sound=True)
    assert etc.n_samples == 2
