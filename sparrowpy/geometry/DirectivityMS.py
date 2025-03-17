"""SoundObject class for spatial audio reproduction."""
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
