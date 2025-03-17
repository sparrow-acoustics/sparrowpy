"""SoundObject class for spatial audio reproduction."""
import matplotlib
import numpy as np
from .DirectivityMS import DirectivityMS


class SoundObject():
    """A class holding the common properties for Source and Receiver."""

    position: np.ndarray
    view: np.ndarray
    up: np.ndarray

    def __init__(
            self, position: np.ndarray, view: np.ndarray,
            up: np.ndarray) -> None:
        """Init a sound object.

        Parameters
        ----------
        position : np.ndarray
            position of the sound object in m
        view : np.ndarray
            view vector of sound object
        up : np.ndarray
            uo vector of sound object

        """
        self.position = np.array(position, dtype=float)
        assert self.position.shape == (3,)
        self.view = np.array(view, dtype=float)
        self.view /= np.sqrt(np.dot(view, view))
        assert self.view.shape == (3,)
        self.up = np.array(up, dtype=float)
        self.up /= np.sqrt(np.dot(up, up))
        assert self.up.shape == (3,)

    def plot(self, ax: matplotlib.axes.Axes, **kwargs):
        """Plot SoundObject position and orientation.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on.
        **kwargs
            Keyword arguments that are passed to
            ``matplotlib.pyplot.scatter()``.

        """
        xyz = self.position
        ax.scatter(xyz[0], xyz[1], xyz[2], kwargs)


class SoundSource(SoundObject):
    """Acoustic sound source inhered from SoundObject."""

    directivity: DirectivityMS
    sound_power: float

    def __init__(
            self, position: np.ndarray, view: np.ndarray,
            up: np.ndarray, directivity: DirectivityMS = None,
            sound_power: float = 1) -> None:
        """Init sound source.

        Parameters
        ----------
        position : np.ndarray
            position of the sound source in m
        view : np.ndarray
            view vector of sound source
        up : np.ndarray
            uo vector of sound source
        directivity : DirectivityMS, optional
            Directivity, by default None
        sound_power : float, optional
            sound power of the source in Watt, by default 1

        """
        super(SoundSource, self).__init__(position, view, up)
        self.sound_power = float(sound_power)
        if directivity is not None:
            assert isinstance(directivity, DirectivityMS)
        self.directivity = directivity

    def plot(self, ax, **kwargs):
        """Plot Source position and orientation.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on.
        **kwargs
            Keyword arguments that are passed to
            ``matplotlib.pyplot.scatter()``.

        """
        super(SoundSource, self).plot(ax, color='r', label='Source', **kwargs)
