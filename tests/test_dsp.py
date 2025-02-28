import sparrowpy as sp
import pyfar as pf
import numpy as np
import pytest

def test_dsp_():
    a = sp.dsp.generate_dirac_impulses(60, 343, 2)
    assert isinstance (a, pf.Signal)
    assert a.sampling_rate