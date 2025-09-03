import numpy as np
import numpy.testing as npt
import pytest
from sparrowpy.classes.RadiosityFast import _bin_patch_energy_to_detector_dirs
from sparrowpy.classes.RadiosityFast import _accumulate_direct_sound_into_bins


### TEST PRIVATE FUNCITONS###
def test_binning_sum_conservation_per_RBS(six_axis_unit):
    """Binning from patches to detector directions conserves energy.

    The sum over detector directions (R,D,B,S → R,B,S) must equal the
    sum over patches (R,P,B,S → R,B,S). This ensures that no energy
    is lost or double-counted in the mapping.
    """
    rng = np.random.default_rng(0)
    R,P,B,S = 2, 7, 3, 5
    rec_xyz       = rng.normal(size=(R,3))
    patch_centers = rng.normal(size=(P,3))
    data_rpbs     = rng.random((R,P,B,S))

    out = _bin_patch_energy_to_detector_dirs(
        rec_xyz=rec_xyz,
        patch_centers=patch_centers,
        det_dirs_unit=six_axis_unit,
        data_rpbs=data_rpbs,
        eps=1e-9,
    )  # (R,D,B,S)

    npt.assert_allclose(out.sum(axis=1), data_rpbs.sum(axis=1), rtol=0, atol=1e-12)


def test_binning_maps_expected_directions(six_axis_unit):
    """Binning assigns patches to the correct detector direction.

    With a receiver at the origin and three patches placed at -x, -y, -z,
    the arrival vectors are +x, +y, +z. Each patch has a distinct constant
    energy signature (1, 2, 3). The test checks that these signatures
    appear only in the matching detector bins (+x=0, +y=2, +z=4) and that
    all other detector bins remain zero.
    """
    rec_xyz       = np.array([[0.0, 0.0, 0.0]])  # (R=1,3)
    patch_centers = np.array([[-1,0,0],[0,-1,0],[0,0,-1]], float)  # -> +x,+y,+z
    B,S = 2,4
    data = np.zeros((1,3,B,S)); data[0,0]=1.0; data[0,1]=2.0; data[0,2]=3.0

    out = _bin_patch_energy_to_detector_dirs(
        rec_xyz, patch_centers, six_axis_unit, data, eps=1e-9
    )  # (1,6,B,S)

    npt.assert_allclose(out[0,0], 1.0)  # +x
    npt.assert_allclose(out[0,2], 2.0)  # +y
    npt.assert_allclose(out[0,4], 3.0)  # +z
    for d in (1,3,5): npt.assert_allclose(out[0,d], 0.0)

def test_accumulate_direct_sound_adds_to_correct_dir_and_sample(six_axis_unit):
    """
    Verify that direct sound energies are inserted into the correct detector
    direction and time sample.

    This test sets up two receivers located along the +x axis from a source at
    the origin. Both receivers should therefore map their direct sound into the
    +x detector bin (index 0). Each receiver is assigned a distinct vector of
    band energies (ds_all) and an arrival time index (tt). After calling
    `_accumulate_direct_sound_into_bins`, the output tensor must contain these
    energies exactly at [receiver, +x direction, :, time_index], with all other
    entries remaining zero.
    """
    R,D,B,S = 2, 6, 3, 16
    out = np.zeros((R,D,B,S))
    rec_xyz = np.array([[1.0,0.0,0.0], [2.0,0.0,0.0]])  # both point +x from src
    src_pos = np.array([0.0,0.0,0.0])
    det_dirs = six_axis_unit
    ds_all   = np.array([[1.0,2.0,3.0],
                         [0.5,0.0,4.0]])  # (R,B)
    tt = np.array([3, 10])               # (R,)

    _accumulate_direct_sound_into_bins(
        out_rdbs=out,
        rec_xyz=rec_xyz,
        src_pos=src_pos,
        det_dirs_unit=det_dirs,
        ds_all_rb=ds_all,
        n_delay_all_r=tt,
        eps=1e-9,
    )

    exp = np.zeros_like(out)
    exp[0, 0, :,  3] += ds_all[0]  # +x is index 0
    exp[1, 0, :, 10] += ds_all[1]
    npt.assert_allclose(out, exp, rtol=0, atol=0.0)

def test_accumulate_direct_sound_raises_on_coincident_src_rec(six_axis_unit):
    '''Raise error when receiver is placed a a ptach'''
    out = np.zeros((1,6,1,8))
    rec_xyz = np.array([[0.0,0.0,0.0]])
    src_pos = np.array([0.0,0.0,0.0])  # coincident → should raise
    det_dirs = six_axis_unit
    ds_all = np.ones((1,1))
    tt = np.array([0])

    with pytest.raises(ValueError, match="source->receiver"):
        _accumulate_direct_sound_into_bins(out, rec_xyz, src_pos, det_dirs, ds_all, tt, eps=1e-9)
