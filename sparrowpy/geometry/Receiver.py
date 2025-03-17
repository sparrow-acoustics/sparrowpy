"""SoundObject class for spatial audio reproduction."""
import matplotlib
import numpy as np
import pyfar as pf
import sofar as sf

from .SoundSource import SoundObject


class Receiver(SoundObject):
    """Receiver object inhered from SoundObject."""

    def __init__(
            self, position: np.ndarray, view: np.ndarray,
            up: np.ndarray) -> None:
        """Init sound receiver.

        Parameters
        ----------
        position : np.ndarray
            cartesian positions for receiver.
        view : np.ndarray
            view vector of sound receiver.
        up : np.ndarray
            up vector of sound receiver.

        """
        super(Receiver, self).__init__(position, view, up)

    def plot(self, ax, **kwargs):
        """Plot Receiver position and orientation.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on.
        **kwargs
            Keyword arguments that are passed to
            ``matplotlib.pyplot.scatter()``.

        """
        super(Receiver, self).plot(ax, color='b', label='Receiver', **kwargs)
