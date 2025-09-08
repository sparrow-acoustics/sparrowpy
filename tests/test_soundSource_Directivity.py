import numpy as np
import sparrowpy.sound_object as so
import os
import numpy.testing as npt


def test_sound_source_init():
    """Test SoundSource initialization."""
    position = np.array([0, 0, 0])
    view = np.array([1, 0, 0])
    up = np.array([0, 0, 1])
    path = os.path.join(
        os.path.dirname(__file__), 'test_data',
        'Genelec8020_DAF_2016_1x1.v17.ms.sofa')
    directivity = so.DirectivityMS(path)
    sound_source = so.SoundSource(position, view, up, directivity)

    # check any direction
    assert isinstance(sound_source, so.SoundSource)
    assert isinstance(sound_source.directivity, so.DirectivityMS)
    res = sound_source.get_directivity(view, 1000)
    npt.assert_almost_equal((2009.3389892578125+0j), res)
    res = sound_source.get_directivity(up, 1000)
    npt.assert_almost_equal((3343.79980469+0j), res)

    # set different view and check if rotation works
    view = np.array([0, 1, 0])
    sound_source = so.SoundSource(position, view, up, directivity)
    res = sound_source.get_directivity(view, 1000)
    npt.assert_almost_equal((2009.3389892578125+0j), res)
    res = sound_source.get_directivity(up, 1000)
    npt.assert_almost_equal((3343.79980469+0j), res)


def test_sound_source_multiple_targets():
    """Test SoundSource multiple targets."""
    position = np.array([0, 0, 0])
    view = np.array([1, 0, 0])
    up = np.array([0, 0, 1])
    path = os.path.join(
        os.path.dirname(__file__), 'test_data',
        'Genelec8020_DAF_2016_1x1.v17.ms.sofa')
    directivity = so.DirectivityMS(path)
    sound_source = so.SoundSource(position, view, up, directivity)

    # check any direction
    assert isinstance(sound_source, so.SoundSource)
    assert isinstance(sound_source.directivity, so.DirectivityMS)
    res = sound_source.get_directivity(np.array((view, up)), 1000)
    npt.assert_almost_equal(res.shape, (2,))
    npt.assert_almost_equal((2009.3389892578125+0j), res[0])
    npt.assert_almost_equal((3343.79980469+0j), res[1])

    # set different view and check if rotation works
    view = np.array([0, 1, 0])
    sound_source = so.SoundSource(position, view, up, directivity)
    res = sound_source.get_directivity(np.array((view, up)), 1000)
    npt.assert_almost_equal((2009.3389892578125+0j), res[0])
    npt.assert_almost_equal((3343.79980469+0j), res[1])
