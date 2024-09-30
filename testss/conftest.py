"""Fixtures for the tests in the tests/ directory."""
import sparapy as sp
import pyfar as pf
import numpy as np
import pytest
import os

@pytest.fixture()
def sample_walls():
    """Return a list of 6 walls, which form a cube."""
    return sp.testing.shoebox_room_stub(1, 1, 1)


@pytest.fixture()
def sofa_data_diffuse():
    """Return a list of 6 walls, which form a cube."""
    gaussian = pf.samplings.sph_gaussian(sh_order=1)
    gaussian = gaussian[gaussian.z>0]
    sources = gaussian.copy()
    receivers = gaussian.copy()
    frequencies = pf.dsp.filter.fractional_octave_frequencies(
        1, (100, 1000))[0]
    data = np.ones((sources.csize, receivers.csize, frequencies.size))
    return (pf.FrequencyData(data, frequencies), sources, receivers)



def pytest_generate_tests(metafunc):
    if "posh" in metafunc.fixturenames:
        r = pf.samplings.sph_equal_angle(30)
        r = r[r.z>0]
        metafunc.parametrize("posh",r.cartesian.tolist())
    if "posj" in metafunc.fixturenames:
        r = pf.samplings.sph_equal_angle(30)
        r = r[r.z>0]
        metafunc.parametrize("posj",r.cartesian.tolist())
        
    
    if "mdist" in metafunc.fixturenames:
        
        li=[]
        
        for file in os.listdir('testss\\brdf_examples\\'):
            
            a = float(file.split("_")[-1].split(".sofa")[0])
            
            if a == 10:
                
                s=float(file.split("brdf_s")[1].split("_")[0])
                
                if s==0:
                    dist = pf.io.read_sofa('testss\\brdf_examples\\'+file)
                    
                    li.append((dist,a,s))
        
        metafunc.parametrize("mdist",li)