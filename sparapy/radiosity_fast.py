"""Module for the radiosity simulation."""
import matplotlib as mpl
import numpy as np
import pyfar as pf
import sofar as sf
from tqdm import tqdm

import sparapy.geometry as geo


class DRadiosityFast():
    """Radiosity object for directional scattering coefficients."""

    _walls_area: np.ndarray
    _walls_points: np.ndarray
    _walls_normal: np.ndarray
    _walls_center: np.ndarray
    _walls_up_vector: np.ndarray
    _patches_area: np.ndarray
    _patches_center: np.ndarray
    _patches_size: np.ndarray
    _patches_points: np.ndarray
    _patches_normal: np.ndarray
    _patch_size: float
    _max_order_k: int

    absorption: np.ndarray
    scattering: np.ndarray

    visibility_matrix: np.ndarray
    n_bins: int
    sound_attenuation_factor: np.ndarray
    patch_to_wall_ids: np.ndarray
    speed_of_sound: float
    form_factors: np.ndarray


    def __init__(
            self, walls_area, walls_points, walls_normal, walls_center,
            walls_up_vector, patches_area, patches_center, patches_size,
            patches_points, patches_normal, patch_size, speed_of_sound,
            max_order_k=None, visibility_matrix=None,):
        """Create a Radiosity object for directional scattering coefficients."""
        self._walls_area = walls_area
        self._walls_points = walls_points
        self._walls_normal = walls_normal
        self._walls_center = walls_center
        self._walls_up_vector = walls_up_vector
        self._patches_area = patches_area
        self._patches_center = patches_center
        self._patches_size = patches_size
        self._patches_points = patches_points
        self._patches_normal = patches_normal
        self._patch_size = patch_size
        self._speed_of_sound = speed_of_sound
        if max_order_k is not None:
            self._max_order_k = max_order_k
        if visibility_matrix is not None:
            self._visibility_matrix = visibility_matrix

    @classmethod
    def from_polygon(
            cls, polygon_list, patch_size,
            speed_of_sound=346.18):
        """Create a Radiosity object for directional scattering coefficients.

        Parameters
        ----------
        polygon_list : list[PatchesDirectional]
            list of patches
        patch_size : float
            maximal patch size in meters.
        max_order_k : int
            max order of energy exchange iterations.
        ir_length_s : float
            length of ir in seconds.
        sofa_path : Path, string, list of Path, list of string
            path of directional scattering coefficients or list of path
            for each Patch.
        speed_of_sound : float, optional
            Speed of sound in m/s, by default 346.18 m/s
        source : SoundSource, optional
            Source object, by default None, can be added later.

        """
        # save wall information
        walls_points = np.array([p.pts for p in polygon_list])
        walls_area = np.array([p.size for p in polygon_list])
        walls_center = np.array([p.center for p in polygon_list])
        walls_normal = np.array([p.normal for p in polygon_list])
        walls_up_vector = np.array([p.up_vector for p in polygon_list])

        # create patches
        patches_points = []
        patch_to_wall_ids = []
        for i, polygon in enumerate(polygon_list):
            patches_points_wall = geo.create_patches(polygon, patch_size)
            patch_to_wall_ids.extend([i for _ in range(len(patches_points_wall))])
            patches_points.extend(patches_points_wall)
        patches_points = np.array(patches_points)
        patch_to_wall_ids = np.array(patch_to_wall_ids)

        # calculate patch information
        patches_size = geo.calculate_size(patches_points)
        patches_area = geo.calculate_area(patches_points)
        patches_center = geo.calculate_center(patches_points)
        patches_normal = np.array([
            walls_normal[i] for i in patch_to_wall_ids])

        # create radiosity object
        return cls(
            walls_area, walls_points, walls_normal, walls_center,
            walls_up_vector, patches_area, patches_center, patches_size,
            patches_points, patches_normal, patch_size, speed_of_sound)

    def calculate_form_factors(self):
        """Calculate the form factors."""
        self._form_factors = form_factor_kang(
            self.patches_center, self.patches_normal,
            self.patches_size)

    @property
    def form_factors(self):
        """Return the form factor."""
        return self._form_factors

    @property
    def walls_area(self):
        """Return the area of the walls."""
        return self._walls_area

    @property
    def walls_points(self):
        """Return the points of the walls."""
        return self._walls_points

    @property
    def walls_normal(self):
        """Return the normal of the walls."""
        return self._walls_normal

    @property
    def walls_center(self):
        """Return the center of the walls."""
        return self._walls_center

    @property
    def walls_up_vector(self):
        """Return the up vector of the walls."""
        return self._walls_up_vector

    @property
    def patches_area(self):
        """Return the area of the patches."""
        return self._patches_area

    @property
    def patches_center(self):
        """Return the center of the patches."""
        return self._patches_center

    @property
    def patches_size(self):
        """Return the size of the patches."""
        return self._patches_size

    @property
    def patches_points(self):
        """Return the points of the patches."""
        return self._patches_points

    @property
    def patches_normal(self):
        """Return the normal of the patches."""
        return self._patches_normal

    @property
    def patch_size(self):
        """Return the size of the patches."""
        return self._patch_size

    @property
    def max_order_k(self):
        """Return the max order of energy exchange iterations."""
        return self._max_order_k


def calculate_init_energy(
        source_position, patches_center, patches_normal,
        patches_size):

    n_patches = patches_center.shape[0]
    idx_l = np.zeros(n_patches)
    idx_l[:] = np.nan
    idx_l[np.abs(patches_normal[:, 2]) > 0.99] = 0
    idx_l[np.abs(patches_normal[:, 1]) > 0.99] = 0
    idx_l[np.abs(patches_normal[:, 0]) > 0.99] = 1
    idx_m = np.zeros(n_patches)
    idx_m[:] = np.nan
    idx_m[np.abs(patches_normal[:, 2]) > 0.99] = 1
    idx_m[np.abs(patches_normal[:, 1]) > 0.99] = 2
    idx_m[np.abs(patches_normal[:, 0]) > 0.99] = 2
    idx_n = np.zeros(n_patches)
    idx_n[:] = np.nan
    idx_n[np.abs(patches_normal[:, 2]) > 0.99] = 2
    idx_n[np.abs(patches_normal[:, 1]) > 0.99] = 1
    idx_n[np.abs(patches_normal[:, 0]) > 0.99] = 0

    dl = patches_center[:, idx_l]
    dm = patches_center[:, idx_m]
    dn = patches_center[:, idx_n]
    dd_l = patches_size[:, idx_l]
    dd_m = patches_size[:, idx_m]
    dd_n = patches_size[:, idx_n]
    S_x = source_position[idx_l]
    S_y = source_position[idx_m]
    S_z = source_position[idx_n]

    half_l = dd_l/2
    half_n = dd_n/2
    half_m = dd_m/2

    sin_phi_delta = (dl + half_l - S_x)/ (np.sqrt(np.square(
        dl+half_l-S_x) + np.square(dm-S_y) + np.square(dn-S_z)))

    k_phi = -1 if dl - half_l <= S_x <= dl + half_l else 1
    sin_phi = k_phi * (dl - half_l - S_x) / (np.sqrt(np.square(
        dl-half_l-S_x) + np.square(dm-S_y) + np.square(dn-S_z)))

    plus  = np.arctan(np.abs((dm+half_m-S_y)/S_z))
    minus = np.arctan(np.abs((dm-half_m-S_y)/S_z))

    k_beta = -1 if (dn - half_n) <= S_z <= (dn + half_n) else 1
    beta = np.abs(plus-(k_beta*minus))

    energy = (np.abs(sin_phi_delta-sin_phi) ) * beta / (4*np.pi)
    distance = np.sqrt(np.square(dl-S_x) + np.square(dm-S_y) + np.square(dn-S_z))
    return (energy, distance)


def check_visibility(
        patches_center:np.ndarray, patches_normal:np.ndarray) -> np.ndarray:
    """Check the visibility between patches."""
    n_patches = patches_center.shape[0]
    visibility_matrix = np.zeros((n_patches, n_patches), dtype=bool)
    for i_source in range(n_patches):
        for i_receiver in range(n_patches):
            patches_parallel = np.abs(np.dot(
                patches_normal[i_source], patches_normal[i_receiver]) -1) < 1e-5
            same_dim = np.sum(
                patches_normal[i_source] * patches_center[i_source]) == np.sum(
                    patches_normal[i_receiver] * patches_center[i_receiver])
            if i_source == i_receiver:
                visibility_matrix[i_source, i_receiver] = False
            elif patches_parallel and same_dim:
                visibility_matrix[i_source, i_receiver] = False
            else:
                visibility_matrix[i_source, i_receiver] = True


def form_factor_kang(
        patches_center:np.ndarray, patches_normal:np.ndarray,
        patches_size:np.ndarray, visibility_matrix:np.ndarray) -> np.ndarray:
    """Calculate the form factors between patches.

    Parameters
    ----------
    patches_center : np.ndarray
        center points of all patches of shape (n_patches, 3)
    patches_normal : np.ndarray
        normal vectors of all patches of shape (n_patches, 3)
    patches_size : np.ndarray
        size of all patches of shape (n_patches, 3)

    Returns
    -------
    form_factors : np.ndarray
        form factors between all patches of shape (n_patches, n_patches)

    """
    n_patches = patches_center.shape[0]
    form_factors = np.zeros((n_patches, n_patches))
    for i_source in range(n_patches):
        source_center = patches_center[i_source]
        source_normal = patches_normal[i_source]
        dd_l = patches_size[i_source, 0]
        dd_m = patches_size[i_source, 1]
        dd_n = patches_size[i_source, 2]
        for i_receiver in range(n_patches):
            receiver_center = patches_center[i_receiver]
            difference = np.abs(receiver_center-source_center)
            # calculation of form factors
            receiver_normal = patches_normal[i_receiver]
            dot_product = np.dot(receiver_normal, source_normal)

            if dot_product == 0:  # orthogonal

                if np.abs(source_normal[0]) > 1e-5:
                    idx_source = set([2, 1])
                    dl = source_center[2]
                    dm = source_center[1]
                elif np.abs(source_normal[1]) > 1e-5:
                    idx_source = set([2, 0])
                    dl = source_center[2]
                    dm = source_center[0]
                elif np.abs(source_normal[2]) > 1e-5:
                    idx_source = set([0, 1])
                    dl = source_center[1]
                    dm = source_center[0]

                if np.abs(receiver_normal[0]) > 1e-5:
                    idx_receiver = set([2, 1])
                    dl_prime = receiver_center[1]
                    dn_prime = receiver_center[2]
                elif np.abs(receiver_normal[1]) > 1e-5:
                    idx_receiver = set([0, 2])
                    dl_prime = receiver_center[0]
                    dn_prime = receiver_center[2]
                elif np.abs(receiver_normal[2]) > 1e-5:
                    idx_receiver = set([0, 1])
                    dl_prime = receiver_center[1]
                    dn_prime = receiver_center[0]
                idx_l = list(idx_receiver.intersection(idx_source))[0]
                idx_s = list(idx_source.difference(set([idx_l])))[0]
                idx_r = list(idx_receiver.difference(set([idx_l])))[0]
                dm = np.abs(
                    source_center[idx_s]-receiver_center[idx_s])
                dl = source_center[idx_l]
                dl_prime = receiver_center[idx_l]
                dn_prime = np.abs(
                    source_center[idx_r]-receiver_center[idx_r])

                d = np.sqrt( ( (dl - dl_prime) ** 2 ) + ( dm ** 2 ) + (
                    dn_prime ** 2) )

                # Equation 13
                A = ( dm - ( 0.5 * dd_m ) ) / ( np.sqrt( ( (
                    dl - dl_prime) ** 2 ) + ( ( dm - (
                        0.5 * dd_m ) ) ** 2 ) + ( dn_prime ** 2) ) )

                # Equation 14
                B_num = dm + (0.5 * dd_m)
                B_denum =  np.sqrt(( np.square(dl - dl_prime) ) + (
                    np.square(dm + (0.5*dd_m)) ) + (
                        np.square(dn_prime)) )
                B = B_num/B_denum

                # Equation 15
                one = np.arctan( np.abs( ( dl - (
                    0.5*dd_l) - dl_prime ) / (dn_prime) ) )
                two = np.arctan( np.abs( ( dl + (
                    0.5*dd_l) - dl_prime ) / (dn_prime) ) )

                k = -1 if np.abs(dl - dl_prime) < 1e-12 else 1

                theta = np.abs( one - (k*two) )

                # Equation 11
                ff =  (
                    1 / (2 * np.pi) ) * (np.abs(
                        (A ** 2) - (B ** 2) )) * theta

            else:
                # parallel
                if difference[0] > 1e-5:
                    dl = receiver_center[1]
                    dm = receiver_center[0]
                    dn = receiver_center[2]
                    dl_prime = source_center[1]
                    dm_prime = source_center[0]
                    dn_prime = source_center[2]
                elif difference[1] > 1e-5:
                    dl = receiver_center[0]
                    dm = receiver_center[1]
                    dn = receiver_center[2]
                    dl_prime = source_center[0]
                    dm_prime = source_center[1]
                    dn_prime = source_center[2]
                elif difference[2] > 1e-5:
                    dl = receiver_center[1]
                    dm = receiver_center[2]
                    dn = receiver_center[0]
                    dl_prime = source_center[1]
                    dm_prime = source_center[2]
                    dn_prime = source_center[0]
                else:
                    raise AssertionError()

                d = np.sqrt(
                    ( (dl - dl_prime) ** 2 ) +
                    ( (dn - dn_prime) ** 2 ) +
                    ( (dm - dm_prime) ** 2 ) )
                # Equation 16
                ff =  ( dd_l * dd_n * ( (
                    dm-dm_prime) ** 2 ) ) / ( np.pi * ( d**4 ) )

            form_factors[i_source, i_receiver] = ff
    return form_factors
