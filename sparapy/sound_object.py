import matplotlib
import numpy as np
import pyfar as pf
import sofar as sf


class DirectivityMS():
    data: pf.FrequencyData
    receivers: pf.Coordinates

    def __init__(self, file_path: str, source_index=0) -> None:
        sofa = sf.read_sofa(file_path)
        if sofa.GLOBAL_SOFAConventions != 'FreeFieldDirectivityTF':
            raise ValueError('convention need to be FreeFieldDirectivityTF')
        sofa = sf.read_sofa(file_path)
        self.data = pf.FrequencyData(
            sofa.Data_Real[source_index, :] + 1j * sofa.Data_Imag[
                source_index, :], sofa.N)
        if sofa.ReceiverPosition_Type == 'spherical':
            pos = sofa.ReceiverPosition.squeeze().T
            pos[0] = (pos[0] + 360) % 360
            self.receivers = pf.Coordinates(
                pos[0], pos[1], pos[2], 'sph', 'top_elev', 'deg')

    def get_directivity(
            self, source_pos: np.ndarray, source_view: np.ndarray,
            source_up: np.ndarray, target_position: np.ndarray,
            i_freq: int) -> float:
        (azimuth_deg, elevation_deg) = _get_metrics(
            source_pos, source_view, source_up, target_position)
        index, _ = self.receivers.find_nearest_k(
            (azimuth_deg+360) % 360, elevation_deg, 1, k=1,
            domain='sph', convention='top_elev', unit='deg')
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

    def plot(self, ax: matplotlib.axes.Axes, color, label):
        xyz = self.position
        ax.scatter(xyz[0], xyz[1], xyz[2], color=color, label=label)



class SoundSource(SoundObject):
    """Acoustic sound source inhered from SoundObject
    """

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

    def plot(self, ax):
        super(SoundSource, self).plot(ax, 'r', 'Source')


class Receiver(SoundObject):
    impulse_response: pf.Signal

    def __init__(
            self, position: np.ndarray, view: np.ndarray,
            up: np.ndarray, impulse_response: pf.Signal = None) -> None:
        super(Receiver, self).__init__(position, view, up)
        if impulse_response is not None:
            assert isinstance(impulse_response, pf.Signal)
        self.impulse_response = impulse_response

    def plot(self, ax):
        super(Receiver, self).plot(ax, 'b', 'Receiver')
