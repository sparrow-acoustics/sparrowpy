"""Test the radiosity_sound_object module."""
import os

import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
import sparrowpy.sound_object as so


def test_directivity_ms():
    """Test DirectivityMS class."""
    path = os.path.join(
        os.path.dirname(__file__), 'test_data', 'ITA_Dodecahedron.sofa')
    directivity = so.DirectivityMS(path)
    assert directivity.data.cshape[0] == directivity.receivers.cshape[0]
    assert isinstance(directivity.data, pf.FrequencyData)
    assert isinstance(directivity.receivers, pf.Coordinates)


def test_get_directivity():
    """Test get_directivity function."""
    path = os.path.join(
        os.path.dirname(__file__), 'test_data', 'ITA_Dodecahedron.sofa')
    directivity = so.DirectivityMS(path)
    target_pos_G = [-1, 0, -2]
    pos_G = [0, 0, 2]
    up_G = [0, 1, 0]
    view_G = [-0.707106781186548, 0, 0.707106781186548]
    fac = directivity.get_directivity(pos_G, view_G, up_G, target_pos_G, 2)
    npt.assert_almost_equal(fac, 0.0001908)


@pytest.mark.parametrize('function', [
    (so.SoundObject),
    (so.SoundSource),
    (so.Receiver) ])
def test_sound_object(function):
    """Test SoundObject class."""
    sound_object = function([0, 0, 0], [2, 0, 0], [0, 0, 2])
    npt.assert_equal(sound_object.position, np.array([0, 0, 0]))
    npt.assert_equal(sound_object.view, np.array([1, 0, 0]))
    npt.assert_equal(sound_object.up, np.array([0, 0, 1]))


@pytest.mark.parametrize('function', [
    (so.SoundObject),
    (so.SoundSource),
    (so.Receiver) ])
@pytest.mark.parametrize(('position', 'view', 'up'), [
    ('a', [1, 0, 0], [0, 0, 1]),
    ([0, 0, 0], [1, 0, 0], 1),
    ([0, 0, 0], ['a'], [0, 0, 1]),
])
def test_sound_object_error(function, position, view, up):
    """Test SoundObject class."""
    with pytest.raises((ValueError, AssertionError)):
        function(position, view, up)


def test_sound_source_defaults():
    """Test SoundSource class."""
    sound_source = so.SoundSource([0, 0, 0], [2, 0, 0], [0, 0, 2])
    npt.assert_equal(sound_source.sound_power, 1)
    npt.assert_equal(sound_source.directivity, None)


@pytest.mark.parametrize(('directivity', 'sound_power'), [
    ('a', 1),
    (None, 'a'),
])
def test_sound_source_errors(directivity, sound_power):
    """Test SoundSource class."""
    with pytest.raises((ValueError, AssertionError)):
        so.SoundSource(
            [0, 0, 0], [2, 0, 0], [0, 0, 2], directivity, sound_power)


def test__get_metrics():
    """Test _get_metrics function."""
    target_pos_G = [-1, 0, -2]
    pos_G = [0, 0, 2]
    up_G = [0, 1, 0]
    view_G = [-0.707106781186548, 0, 0.707106781186548]
    (azimuth_deg, elevation_deg) = so._get_metrics(
        pos_G, view_G, up_G, target_pos_G)
    assert azimuth_deg == -120.96375653207352
    assert elevation_deg == 0
