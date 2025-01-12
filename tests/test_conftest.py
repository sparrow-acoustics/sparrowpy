import sofar as sf
import numpy.testing as npt
import numpy as np


def test_brdf_s_0(brdf_s_0):
    sofa = sf.read_sofa(brdf_s_0)
    for i in range(4):
        npt.assert_almost_equal(sofa.Data_Real[i - 2, i], 1.1026578)
        npt.assert_almost_equal(
            np.sum(sofa.Data_Real[i - 2]), 1.1026578)


def test_brdf_s_1(brdf_s_1):
    sofa = sf.read_sofa(brdf_s_1)
    npt.assert_almost_equal(sofa.Data_Real, 1 / np.pi)
