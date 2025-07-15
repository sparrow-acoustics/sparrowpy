"""SoundObject class for spatial audio reproduction."""
import matplotlib
import numpy as np
import pyfar as pf
import sofar as sf


class DirectivityMS():
    """Directivity class for FreeFieldDirectivityTF convention."""

    data: pf.FrequencyData
    receivers: pf.Coordinates

    def __init__(self, file_path: str, source_index=0) -> None:
        """Init DirectivityMS.

        Parameters
        ----------
        file_path : str
            directivity path for sofa file.
        source_index : int, optional
            source index of directivity, by default 0

        """
        sofa = sf.read_sofa(file_path, verbose=False)
        self.data = pf.FrequencyData(
            sofa.Data_Real[source_index, :] + 1j * sofa.Data_Imag[
                source_index, :], sofa.N)
        positions = sofa.ReceiverPosition
        if positions.ndim > 2:
            positions = np.squeeze(positions)
        self.receivers = pf.io.io._sofa_pos(
            sofa.ReceiverPosition_Type, positions)
        if  self.receivers.cdim != 1:
            raise ValueError(
                'DirectivityMS only supports 1D coordinates, '
                f'got {self.receivers.cdim}D coordinates. Squeezing did not '
                'work.')

    def get_directivity(
            self, source_pos: np.ndarray, source_view: np.ndarray,
            source_up: np.ndarray, target_position: np.ndarray,
            i_freq: int) -> float:
        """Get Directivity for certain position.

        Parameters
        ----------
        source_pos : np.ndarray
            cartesian source position in m
        source_view : np.ndarray
            cartesian source view in m
        source_up : np.ndarray
            cartesian source up in m
        target_position : np.ndarray
            cartesian target position in m
        i_freq : int
            frequency bin index

        Returns
        -------
        float
            nearest directivity factor for given position and orientation.

        """
        (azimuth_deg, elevation_deg) = _get_metrics(
            source_pos, source_view, source_up, target_position)
        find = pf.Coordinates.from_spherical_elevation(
            azimuth_deg/180*np.pi, elevation_deg/180*np.pi, 1,
        )
        index, _ = self.receivers.find_nearest(find)
        return self.data.freq[index, i_freq]


def _get_metrics(pos_G, view_G, up_G, target_pos_G):
    pos_G = np.array(pos_G, dtype=float)
    view_G = np.array(view_G, dtype=float)
    up_G = np.array(up_G, dtype=float)
    target_pos_G = np.array(target_pos_G, dtype=float)
    direction_G = target_pos_G - pos_G

    x_dash = np.cross(view_G, up_G)
    y_dash = up_G
    z_dash = -view_G

    w_x_local = np.dot(direction_G, x_dash)
    w_y_local = np.dot(direction_G, y_dash)
    w_z_local = np.dot(direction_G, z_dash)

    azimuth_deg = np.arctan2(-w_x_local, -w_z_local) / np.pi * 180
    w = np.array([w_x_local, w_y_local, w_z_local])
    w_y_local_normalized = w_y_local / np.sqrt(np.dot(w, w))
    elevation_deg = np.arcsin(w_y_local_normalized) / np.pi * 180

    return (azimuth_deg, elevation_deg)


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

    def get_directivity(
            self, target_position: np.ndarray, frequency: float) -> float:
        """Get Directivity for certain position and frequency.

        Parameters
        ----------
        target_position : np.ndarray
            cartesian target position in m of shape (3,) or (n, 3).
        frequency : float
            frequency in Hz.

        Returns
        -------
        float
            nearest directivity factor for given position and frequency.
        """
        i_freq = np.argmin(np.abs(self.directivity.data.frequencies-frequency))
        if target_position.size == 3:
            return self.directivity.get_directivity(
                self.position, self.view, self.up, target_position, i_freq)
        else:
            return np.array([
                self.directivity.get_directivity(
                    self.position, self.view, self.up, pos, i_freq)
                for pos in target_position
            ])[:, 0]

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
