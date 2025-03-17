"""Module for the geometry of the room and the environment."""
import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt
from .Polygon import Polygon
from .SoundSource import SoundSource
from .Receiver import Receiver


class Environment():
    """Define a Geometry with Walls, Source and Receiver."""

    speed_of_sound: float
    polygons: list[Polygon]
    source: SoundSource
    receiver: Receiver

    def __init__(
            self, polygons: list[Polygon],
            source: SoundSource,
            receiver: Receiver,
            speed_of_sound: float) -> None:
        """Define environment with acoustic Objects and speed of sound.

        Parameters
        ----------
        polygons : list[Polygon]
            input polygons as a list
        source : SoundSource
            sound source in the scene
        receiver : Receiver
            receiver in the scene
        speed_of_sound : float, optional
            speed of sound in m/s

        """
        self.speed_of_sound = speed_of_sound
        self.polygons = polygons
        self.source = source
        self.receiver = receiver

    def plot(self, ax: matplotlib.axes.Axes = None):
        """Plot the environment."""
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        i = 0
        self.source.plot(ax)
        self.receiver.plot(ax)
        for i in range(len(self.polygons)):
            self.polygons[i].plot(ax, colors[i])
        ax.set_xlabel('x',labelpad=30)
        ax.set_ylabel('y',labelpad=30)
        ax.set_zlabel('z')
        ax.set_aspect('equal', 'box')


def _cmp_floats(a, b, atol=1e-12):
    return abs(a-b) < atol


def _magnitude(vector):
    return np.sqrt(np.dot(np.array(vector), np.array(vector)))


def _norm(vector):
    return np.array(vector)/_magnitude(np.array(vector))
