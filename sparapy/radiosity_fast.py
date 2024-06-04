"""Module for the radiosity simulation."""
import numba
import os
import numpy as np
import pyfar as pf


class DRadiosityFast():
    """Radiosity object for directional scattering coefficients."""

    _walls_points: np.ndarray
    _walls_normal: np.ndarray
    _walls_up_vector: np.ndarray
    _patches_points: np.ndarray
    _patches_normal: np.ndarray
    _patch_size: float
    _max_order_k: int
    _n_patches: int
    _speed_of_sound: float

    _visibility_matrix: np.ndarray
    _form_factors: np.ndarray

    # general data for material data
    _n_bins: int
    _frequencies = np.ndarray
    # absorption data
    _absorption: np.ndarray
    _absorption_index: np.ndarray
    _scattering: np.ndarray
    _scattering_index: np.ndarray
    _sources: list[pf.Coordinates]
    _receivers: list[pf.Coordinates]

    _air_attenuation: np.ndarray


    def __init__(
            self, walls_points, walls_normal, walls_up_vector,
            patches_points, patches_normal, patch_size, n_patches,
            patch_to_wall_ids, max_order_k=None,
            visibility_matrix=None,
            form_factors=None):
        """Create a Radiosity object for directional scattering coefficients."""
        self._walls_points = walls_points
        self._walls_normal = walls_normal
        self._walls_up_vector = walls_up_vector
        self._patches_points = patches_points
        self._patches_normal = patches_normal
        self._patch_size = patch_size
        self._n_patches = n_patches
        self._patch_to_wall_ids = patch_to_wall_ids
        if max_order_k is not None:
            self._max_order_k = max_order_k
        if visibility_matrix is not None:
            self._visibility_matrix = visibility_matrix
        if form_factors is not None:
            self._form_factors = form_factors
        self._n_bins = None
        self._frequencies = None
        self._sources = None
        self._receivers = None
        self._absorption = None
        self._wrote_energy = False

    @classmethod
    def from_polygon(
            cls, polygon_list, patch_size):
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
        source : SoundSource, optional
            Source object, by default None, can be added later.

        """
        # save wall information
        walls_points = np.array([p.pts for p in polygon_list])
        walls_normal = np.array([p.normal for p in polygon_list])
        walls_up_vector = np.array([p.up_vector for p in polygon_list])

        # create patches
        (
            patches_points, patches_normal,
            n_patches, patch_to_wall_ids) = process_patches(
            walls_points, walls_normal, patch_size, len(polygon_list))
        # create radiosity object
        return cls(
            walls_points, walls_normal, walls_up_vector,
            patches_points, patches_normal, patch_size, n_patches,
            patch_to_wall_ids)

    def init_energy(self, source_position):
        """Calculate the initial energy."""
        # write or read energy to file
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dir_path, 'energy_exchange.far')
        if self._wrote_energy:
            data = pf.io.read(file_path)
            self.energy_exchange = data['energy_exchange']
        else:
            pf.io.write(file_path, energy_exchange=self.energy_exchange)
            self._wrote_energy = True
        # calculate initial energy
        energy_0, distance_0 = calculate_init_energy(
            source_position, self.patches_center, self.patches_normal,
            self.patches_size)
        air_attenuation = self._air_attenuation
        n_patches = self.n_patches

        for j in range(n_patches):
            if air_attenuation is not None:
                air_attenuation_factor = np.exp(
                    -air_attenuation * distance_0[j])
            else:
                air_attenuation_factor = 1
            self.energy_exchange[0, 0, j, :-1] = energy_0[j] * (
                air_attenuation_factor)
            self.energy_exchange[0, 0, j, -1] = distance_0[j]

        if self.energy_exchange.shape[0] <= 1:
            return None
        patch_to_wall_ids = self._patch_to_wall_ids
        patches_center = self.patches_center
        absorption = self._absorption
        absorption_index = self._absorption_index
        sources = self._sources
        receivers = self._receivers
        scattering = self._scattering
        scattering_index = self._scattering_index
        form_factors = self.form_factors
        visibility_matrix = self.visibility_matrix
        n_combinations = np.sum(visibility_matrix)
        pairs = np.empty((n_combinations, 2), dtype=np.int32)
        i_counter = 0
        for i_source in range(n_patches):
            for i_receiver in range(n_patches):
                if not visibility_matrix[i_source, i_receiver]:
                    continue
                pairs[i_counter, 0] = i_source
                pairs[i_counter, 1] = i_receiver
                i_counter += 1
        for ii in range(n_combinations):
            for jj in range(2):
                if jj == 0:
                    i = pairs[ii, 0]
                    j = pairs[ii, 1]
                else:
                    j = pairs[ii, 0]
                    i = pairs[ii, 1]
                difference_source = source_position - patches_center[i]
                difference_source = pf.Coordinates(
                    difference_source.T[0],
                    difference_source.T[1],
                    difference_source.T[2])
                wall_id_i = patch_to_wall_ids[i]
                difference_receiver = patches_center[i]-patches_center[j]
                distance = np.linalg.norm(difference_receiver)
                difference_receiver = pf.Coordinates(
                    difference_receiver.T[0],
                    difference_receiver.T[1],
                    difference_receiver.T[2])

                if air_attenuation is not None:
                    air_attenuation_factor = np.exp(-air_attenuation * distance)
                else:
                    air_attenuation_factor = 1
                absorption_factor = 1-absorption[absorption_index[wall_id_i]]
                source_idx = sources[wall_id_i].find_nearest(
                    difference_source)[0][0]
                receiver_idx = receivers[wall_id_i].find_nearest(
                    difference_receiver)[0][0]
                ff = form_factors[i, j] if i < j else form_factors[j, i]

                scattering_factor = scattering[scattering_index[wall_id_i]][
                    source_idx, receiver_idx, :]
                self.energy_exchange[1, i, j, :-1] = self.energy_exchange[
                    0, 0, j, :-1] * ff * (
                    air_attenuation_factor * absorption_factor * scattering_factor)
                self.energy_exchange[1, i, j, -1] = distance_0[i]+distance
        # calculate energy exchange
        if self.energy_exchange.shape[0] > 2:
            for j in range(n_patches):
                self.energy_exchange[2:, :, j, :-1] *= self.energy_exchange[
                    1, :, j, :-1]
                self.energy_exchange[2:, :, j, -1] += self.energy_exchange[
                    1, :, j, -1]


    def collect_energy_receiver(
            self, receiver_pos, histogram_time_resolution=1e-3,
            histogram_time_length=1, speed_of_sound=346.18):
        """Collect the energy at the receiver."""
        histogram = np.zeros(
            ((int(histogram_time_length/histogram_time_resolution)),
             self.n_bins))
        patches_center = self.patches_center
        patches_normal = self.patches_normal
        air_attenuation = self._air_attenuation
        energy_exchange = self.energy_exchange
        idx = np.array(np.where(energy_exchange[:, :, :, -1] > 0))
        for ii in range(idx.shape[1]):
            k = idx[0, ii]
            i = idx[1, ii]
            j = idx[2, ii]

            source_pos = patches_center[j]
            R = np.linalg.norm(receiver_pos-source_pos)
            e = energy_exchange[k, i, j, :-1]
            d = energy_exchange[k, i, j, -1]+R
            samples_delay = int(d/speed_of_sound/histogram_time_resolution)

            cos_xi = np.abs(np.sum(
                patches_normal[j, :]*np.abs(receiver_pos-source_pos))) / R

            # Equation 20
            receiver_factor = cos_xi * (np.exp(-air_attenuation*R)) / (
                np.pi * R**2)
            histogram[samples_delay, :] += e*receiver_factor

        return histogram


    def check_visibility(self):
        """Check the visibility between patches."""
        self._visibility_matrix = check_visibility(
            self.patches_center, self.patches_normal)

    def calculate_form_factors(self, method='kang'):
        """Calculate the form factors.

        Parameters
        ----------
        method : str, optional
            _description_, by default 'kang'

        """
        if method == 'kang':
            self._form_factors = form_factor_kang(
                self.patches_center, self.patches_normal,
                self.patches_size, self.visibility_matrix)

    def calculate_form_factors_directivity(self):
        """Calculate the form factors with directivity."""
        sources_array = np.array([s.cartesian for s in self._sources])
        receivers_array = np.array([s.cartesian for s in self._receivers])
        self._form_factors_tilde = _form_factors_with_directivity(
            self.visibility_matrix, self.form_factors, self.n_bins, self.patches_center,
            self._air_attenuation, np.array(self._absorption), self._absorption_index,
            self._patch_to_wall_ids, np.array(self._scattering), self._scattering_index,
            sources_array, receivers_array)

    def calculate_energy_exchange(self, max_order_k: int):
        """Calculate the energy exchange.

        Parameters
        ----------
        max_order_k : int
            maximal order of energy exchange iterations.

        """
        if max_order_k <= 1:
            self.energy_exchange = np.zeros(
                (max_order_k+1, self.n_patches, self.n_patches, self.n_bins+1))
        else:
            self.energy_exchange = _calculate_energy_exchange(
                self._form_factors_tilde, self.patches_center,
                max_order_k, self.n_patches, self.n_bins)

    def _check_set_frequency(self, frequencies:np.ndarray):
        """Check if the frequency data matches the radiosity object."""
        if self._n_bins is None:
            self._n_bins = frequencies.size
        else:
            assert self._n_bins == frequencies.size, \
                "Number of bins do not match"
        if self._frequencies is None:
            self._frequencies = frequencies
        else:
            assert (self._frequencies == frequencies).all(), \
                "Frequencies do not match"

    def set_wall_absorption(self, wall_indexes, absorption:pf.FrequencyData):
        """Set the wall absorption.

        Parameters
        ----------
        wall_indexes : list[int]
            list of walls for the scattering data
        absorption : pf.FrequencyData
            scattering data of cshape (1, )

        """
        self._check_set_frequency(absorption.frequencies)
        if self._absorption is None:
            self._absorption_index = np.empty((self.n_walls), dtype=np.int64)
            self._absorption_index.fill(-1)
            self._absorption = []

        self._absorption.append(absorption.freq.squeeze())
        self._absorption_index[wall_indexes] = len(self._absorption)-1


    def set_air_attenuation(self, air_attenuation:pf.FrequencyData):
        """Set air attenuation factor in Np/m.

        Parameters
        ----------
        air_attenuation : pf.FrequencyData
            Air attenuation factor in Np/m.

        """
        self._check_set_frequency(air_attenuation.frequencies)
        self._air_attenuation = air_attenuation.freq.squeeze()

    def set_wall_scattering(
            self, wall_indexes:list[int],
            scattering:pf.FrequencyData, sources:pf.Coordinates,
            receivers:pf.Coordinates):
        """Set the wall scattering.

        Parameters
        ----------
        wall_indexes : list[int]
            list of walls for the scattering data
        scattering : pf.FrequencyData
            scattering data of cshape (n_sources, n_receivers)
        sources : pf.Coordinates
            source coordinates
        receivers : pf.Coordinates
            receiver coordinates

        """
        assert (sources.z >= 0).all(), \
            "Sources must be in the positive half space"
        assert (receivers.z >= 0).all(), \
            "Receivers must be in the positive half space"
        self._check_set_frequency(scattering.frequencies)
        if self._sources is None:
            self._sources = np.empty((self.n_walls), dtype=pf.Coordinates)
            self._receivers = np.empty((self.n_walls), dtype=pf.Coordinates)
            self._scattering_index = np.empty((self.n_walls), dtype=np.int64)
            self._scattering_index.fill(-1)
            self._scattering = []

        for i in wall_indexes:
            sources_rot, receivers_rot = self._rotate_coords_to_normal(
                self.walls_normal[i], self.walls_up_vector[i],
                sources, receivers)
            self._sources[i] = sources_rot
            self._receivers[i] = receivers_rot

        self._scattering.append(scattering.freq)
        self._scattering_index[wall_indexes] = len(self._scattering)-1

    def _rotate_coords_to_normal(
            self, wall_normal:np.ndarray, wall_up_vector:np.ndarray,
            sources:pf.Coordinates, receivers:pf.Coordinates):
        """Rotate the coordinates to the normal vector."""
        o1 = pf.Orientations.from_view_up(
            wall_normal, wall_up_vector)
        o2 = pf.Orientations.from_view_up([0, 0, 1], [1, 0, 0])
        o_diff = o1.inv()*o2
        euler = o_diff.as_euler('xyz', True).flatten()
        receivers_cp = receivers.copy()
        receivers_cp.rotate('xyz', euler)
        receivers_cp.radius = 1
        sources_cp = sources.copy()
        sources_cp.rotate('xyz', euler)
        sources_cp.radius = 1
        return sources_cp, receivers_cp

    @property
    def n_bins(self):
        """Return the number of frequency bins."""
        return self._n_bins

    @property
    def n_walls(self):
        """Return the number of walls."""
        return self._walls_points.shape[0]

    @property
    def n_patches(self):
        """Return the total number of patches."""
        return self._n_patches

    @property
    def form_factors(self):
        """Return the form factor."""
        return self._form_factors

    @property
    def visibility_matrix(self):
        """Return the visibility matrix."""
        return self._visibility_matrix

    @property
    def walls_area(self):
        """Return the area of the walls."""
        return _calculate_area(self._walls_points)

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
        return _calculate_center(self._walls_points)

    @property
    def walls_up_vector(self):
        """Return the up vector of the walls."""
        return self._walls_up_vector

    @property
    def patches_area(self):
        """Return the area of the patches."""
        return _calculate_area(self._patches_points)

    @property
    def patches_center(self):
        """Return the center of the patches."""
        return _calculate_center(self._patches_points)

    @property
    def patches_size(self):
        """Return the size of the patches."""
        return _calculate_size(self._patches_points)

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

    @property
    def speed_of_sound(self):
        """Return the speed of sound in m/s."""
        return self._speed_of_sound

# @numba.jit(nopython=True)
def _form_factors_with_directivity(
        visibility_matrix, form_factors, n_bins, patches_center, air_attenuation,
        absorption, absorption_index, patch_to_wall_ids,
        scattering, scattering_index, sources, receivers):
    """Calculate the form factors with directivity."""
    n_patches = patches_center.shape[0]
    form_factors_tilde = np.zeros((n_patches, n_patches, n_patches, n_bins))
    # loop over previous patches, current and next patch

    for ii in range(n_patches**3):
        h = ii % n_patches
        i = int(ii/n_patches) % n_patches
        j = int(ii/n_patches**2) % n_patches
        visible_hi = visibility_matrix[
            h, i] if h < i else visibility_matrix[i, h]
        visible_ij = visibility_matrix[
            i, j] if i < j else visibility_matrix[j, i]
        if visible_hi and visible_ij:
            difference_source = patches_center[h]-patches_center[i]
            difference_receiver = patches_center[i]-patches_center[j]
            wall_id_i = int(patch_to_wall_ids[i])
            difference_source /= np.linalg.norm(difference_source)
            difference_receiver /= np.linalg.norm(difference_receiver)
            ff = form_factors[i, j] if i<j else form_factors[j, i] # * patch_j.Area / patch_i.Area # This needs to be included somehow

            distance = np.linalg.norm(difference_receiver)
            if air_attenuation is not None:
                air_attenuation_factor = np.exp(-air_attenuation * distance)
            else:
                air_attenuation_factor = 0
            source_wall_idx = absorption_index[wall_id_i]
            absorption_factor = 1-absorption[source_wall_idx]
            source_idx = np.argmin(np.linalg.norm(
                sources[wall_id_i]-difference_source, axis=-1))
            receiver_idx = np.argmin(np.linalg.norm(
                receivers[wall_id_i]-difference_receiver, axis=-1))
            scattering_factor = scattering[scattering_index[wall_id_i],
                source_idx, receiver_idx, :]
            form_factors_tilde[h, i, j, :] = ff * (
                air_attenuation_factor * absorption_factor * scattering_factor)
    return form_factors_tilde


@numba.jit(nopython=True)
def process_patches(
        polygon_points_array: np.ndarray,
        walls_normal: np.ndarray,
        patch_size:float, n_walls:int):
    """Process the patches.

    Parameters
    ----------
    polygon_points_array : np.ndarray
        points of the polygon of shape (n_walls, 4, 3)
    walls_normal : np.ndarray
        wall normal of shape (n_walls, 3)
    patch_size : float
        maximal patch size in meters of shape (n_walls, 3).
    n_walls : int
        number of walls

    Returns
    -------
    patches_points : np.ndarray
        points of all patches of shape (n_patches, 4, 3)
    patches_normal : np.ndarray
        normal of all patches of shape (n_patches, 3)
    n_patches : int
        number of patches

    """
    n_patches = 0
    n_walls = polygon_points_array.shape[0]
    for i in range(n_walls):
        n_patches += total_number_of_patches(
            polygon_points_array[i, :, :], patch_size)
    patches_points = np.empty((n_patches, 4, 3))
    patch_to_wall_ids = np.empty((n_patches), dtype=np.int64)
    patches_per_wall = np.empty((n_walls), dtype=np.int64)

    for i in range(n_walls):
        polygon_points = polygon_points_array[i, :]
        patches_points_wall = _create_patches(
            polygon_points, patch_size)
        patches_per_wall[i] = patches_points_wall.shape[0]
        j_start = (np.sum(patches_per_wall[:i])) if i > 0 else 0
        j_end = np.sum(patches_per_wall[:i+1])
        patch_to_wall_ids[j_start:j_end] = i
        patches_points[j_start:j_end, :, :] = patches_points_wall
    n_patches = patches_points.shape[0]

    # calculate patch information
    patches_normal = walls_normal[patch_to_wall_ids, :]
    return (patches_points, patches_normal, n_patches, patch_to_wall_ids)


@numba.jit(nopython=True)
def total_number_of_patches(polygon_points:np.ndarray, max_size: float):
    """Calculate the total number of patches.

    Parameters
    ----------
    polygon_points : np.ndarray
        points of the polygon of shape (4, 3)
    max_size : float
        maximal patch size in meters

    Returns
    -------
    n_patches : int
        number of patches

    """
    size = np.empty(polygon_points.shape[1])
    for i in range(polygon_points.shape[1]):
        size[i] = polygon_points[:, i].max() - polygon_points[:, i].min()
    patch_nums = np.array([int(n) for n in size/max_size])
    if patch_nums[2] == 0:
        x_idx = 0
        y_idx = 1
    if patch_nums[1] == 0:
        x_idx = 0
        y_idx = 2
    if patch_nums[0] == 0:
        x_idx = 1
        y_idx = 2

    return patch_nums[x_idx]*patch_nums[y_idx]

# @numba.jit(parallel=True)
def calculate_init_energy(
        source_position: np.ndarray, patches_center: np.ndarray,
        patches_normal: np.ndarray, patches_size: float):
    """Calculate the initial energy from the source.

    Parameters
    ----------
    source_position : np.ndarray
        source position of shape (3,)
    patches_center : np.ndarray
        center of all patches of shape (n_patches, 3)
    patches_normal : np.ndarray
        normal of all patches of shape (n_patches, 3)
    patches_size : float
        size of all patches of shape (n_patches, 3)

    Returns
    -------
    energy : np.ndarray
        energy of all patches of shape (n_patches)
    distance : np.ndarray
        corresponding distance of all patches of shape (n_patches)

    """
    n_patches = patches_center.shape[0]
    energy = np.empty((n_patches, ))
    distance_out = np.empty((n_patches, ))
    for j in  numba.prange(n_patches):
        source_pos = source_position.copy()
        receiver_pos = patches_center[j, :].copy()
        receiver_normal = patches_normal[j, :].copy()
        receiver_size = patches_size[j, :].copy()

        # array([0., 1., 0.]), array([ 0., -1.,  0.]),
        # array([0., 0., 1.]), array([ 0.,  0., -1.]),
        # array([1., 0., 0.]), array([-1.,  0.,  0.])]
        if np.abs(receiver_normal[2]) > 0.99:
            i = 2
            indexes = [0, 1, 2]
        elif np.abs(receiver_normal[1]) > 0.99:
            indexes = [2, 0, 1]
            i = 1
        elif np.abs(receiver_normal[0]) > 0.99:
            i = 0
            indexes = [1, 2, 0]
        else:
            raise AssertionError()
        offset = receiver_pos[i]
        source_pos[i] = np.abs(source_pos[i] - offset)
        receiver_pos[i] = np.abs(receiver_pos[i] - offset)
        dl = receiver_pos[indexes[0]]
        dm = receiver_pos[indexes[1]]
        dn = receiver_pos[indexes[2]]
        dd_l = receiver_size[indexes[0]]
        dd_m = receiver_size[indexes[1]]
        dd_n = receiver_size[indexes[2]]
        S_x = source_pos[indexes[0]]
        S_y = source_pos[indexes[1]]
        S_z = source_pos[indexes[2]]

        half_l = dd_l/2
        half_n = dd_n/2
        half_m = dd_m/2

        sin_phi_delta = (dl + half_l - S_x)/ (np.sqrt(np.square(
            dl+half_l-S_x) + np.square(dm-S_y) + np.square(dn-S_z)))

        k_phi = -1 if np.abs(dl - half_l - S_x) <= 1e-12 else 1
        sin_phi = k_phi * (dl - half_l - S_x) / (np.sqrt(np.square(
            dl-half_l-S_x) + np.square(dm-S_y) + np.square(dn-S_z)))

        plus  = np.arctan(np.abs((dm+half_m-S_y)/S_z))
        minus = np.arctan(np.abs((dm-half_m-S_y)/S_z))

        test1 = dl-half_l-S_x <= 1e-12
        test2 = S_x -dl-half_l <= 1e-12
        k_beta = -1 if test1 and test2 else 1
        beta = np.abs(plus-(k_beta*minus))

        energy[j] = (np.abs(sin_phi_delta-sin_phi) ) * beta / (4*np.pi)
        distance_out[j] = np.sqrt(
            np.square(dl-S_x) + np.square(dm-S_y) + np.square(dn-S_z))

        # energy[j] = sp.radiosity._init_energy_exchange(
        #     dl, dm, dn, dd_l, dd_m, dd_n, S_x, S_y, S_z,
        #     1, np.array([0]), distance_out[j],
        #     np.array([0]), 1)
    return (energy, distance_out)


@numba.jit(nopython=True, parallel=True)
def check_visibility(
        patches_center:np.ndarray, patches_normal:np.ndarray) -> np.ndarray:
    """Check the visibility between patches.

    Parameters
    ----------
    patches_center : np.ndarray
        center points of all patches of shape (n_patches, 3)
    patches_normal : np.ndarray
        normal vectors of all patches of shape (n_patches, 3)

    Returns
    -------
    visibility_matrix : np.ndarray
        boolean matrix of shape (n_patches, n_patches) with True if patches
        can see each other, otherwise false

    """
    n_patches = patches_center.shape[0]
    visibility_matrix = np.empty((n_patches, n_patches), dtype=np.bool_)
    visibility_matrix.fill(False)
    indexes = []
    for i_source in range(n_patches):
        for i_receiver in range(n_patches):
            if i_source < i_receiver:
                indexes.append((i_source, i_receiver))
    indexes = np.array(indexes)
    for i in numba.prange(indexes.shape[0]):
        i_source = indexes[i, 0]
        i_receiver = indexes[i, 1]
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
    return visibility_matrix


@numba.jit(nopython=True, parallel=True)
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
    visibility_matrix : np.ndarray
        boolean matrix of shape (n_patches, n_patches) with True if patches
        can see each other, otherwise false

    Returns
    -------
    form_factors : np.ndarray
        form factors between all patches of shape (n_patches, n_patches)
        note that just i_source < i_receiver are calculated ff[i, j] = ff[j, i]

    """
    n_patches = patches_center.shape[0]
    form_factors = np.zeros((n_patches, n_patches))
    n_combinations = np.sum(visibility_matrix)
    pairs = np.empty((n_combinations, 2), dtype=np.int32)
    i_counter = 0
    for i_source in range(n_patches):
        for i_receiver in range(n_patches):
            if not visibility_matrix[i_source, i_receiver]:
                continue
            pairs[i_counter, 0] = i_source
            pairs[i_counter, 1] = i_receiver
            i_counter += 1
    for i in numba.prange(n_combinations):
        i_source = int(pairs[i, 0])
        i_receiver = int(pairs[i, 1])
        source_center = patches_center[i_source]
        source_normal = patches_normal[i_source]
        receiver_center = patches_center[i_receiver]
        # calculation of form factors
        receiver_normal = patches_normal[i_receiver]
        dot_product = np.dot(receiver_normal, source_normal)

        if dot_product == 0:  # orthogonal

            if np.abs(source_normal[0]) > 1e-5:
                idx_source = set([2, 1])
                dl = source_center[2]
                dm = source_center[1]
                dd_l = patches_size[i_source, 2]
                dd_m = patches_size[i_source, 1]
            elif np.abs(source_normal[1]) > 1e-5:
                idx_source = set([2, 0])
                dl = source_center[2]
                dm = source_center[0]
                dd_l = patches_size[i_source, 2]
                dd_m = patches_size[i_source, 0]
            elif np.abs(source_normal[2]) > 1e-5:
                idx_source = set([0, 1])
                dl = source_center[1]
                dm = source_center[0]
                dd_l = patches_size[i_source, 1]
                dd_m = patches_size[i_source, 0]

            if np.abs(receiver_normal[0]) > 1e-5:
                idx_l = 1 if 1 in idx_source else 2
                idx_s = 0
                idx_r = 2 if 1 in idx_source else 1
                dl_prime = receiver_center[1]
                dn_prime = receiver_center[2]
            elif np.abs(receiver_normal[1]) > 1e-5:
                idx_l = 0 if 0 in idx_source else 2
                idx_s = 1
                idx_r = 2 if 0 in idx_source else 0
                dl_prime = receiver_center[0]
                dn_prime = receiver_center[2]
            elif np.abs(receiver_normal[2]) > 1e-5:
                idx_l = 0 if 0 in idx_source else 1
                idx_s = 2
                idx_r = 1 if 0 in idx_source else 0
                dl_prime = receiver_center[1]
                dn_prime = receiver_center[0]

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
            if np.abs(receiver_normal[0]) > 1e-5:
                dl = receiver_center[1]
                dm = receiver_center[0]
                dn = receiver_center[2]
                dl_prime = source_center[1]
                dm_prime = source_center[0]
                dn_prime = source_center[2]
                dd_l = patches_size[i_source, 1]
                dd_n = patches_size[i_source, 2]
            elif np.abs(receiver_normal[1]) > 1e-5:
                dl = receiver_center[0]
                dm = receiver_center[1]
                dn = receiver_center[2]
                dl_prime = source_center[0]
                dm_prime = source_center[1]
                dn_prime = source_center[2]
                dd_l = patches_size[i_source, 0]
                dd_n = patches_size[i_source, 2]
            elif np.abs(receiver_normal[2]) > 1e-5:
                dl = receiver_center[1]
                dm = receiver_center[2]
                dn = receiver_center[0]
                dl_prime = source_center[1]
                dm_prime = source_center[2]
                dn_prime = source_center[0]
                dd_l = patches_size[i_source, 1]
                dd_n = patches_size[i_source, 0]

            d = np.sqrt(
                ( (dl - dl_prime) ** 2 ) +
                ( (dn - dn_prime) ** 2 ) +
                ( (dm - dm_prime) ** 2 ) )
            # Equation 16
            ff =  ( dd_l * dd_n * ( (
                dm-dm_prime) ** 2 ) ) / ( np.pi * ( d**4 ) )

        form_factors[i_source, i_receiver] = ff
    return form_factors


@numba.jit(nopython=True)
def _create_patches(polygon_points:np.ndarray, max_size):
    """Create patches from a polygon."""
    size = np.empty(polygon_points.shape[1])
    for i in range(polygon_points.shape[1]):
        size[i] = polygon_points[:, i].max() - polygon_points[:, i].min()
    patch_nums = np.array([int(n) for n in size/max_size])
    real_size = size/patch_nums
    if patch_nums[2] == 0:
        x_idx = 0
        y_idx = 1
    if patch_nums[1] == 0:
        x_idx = 0
        y_idx = 2
    if patch_nums[0] == 0:
        x_idx = 1
        y_idx = 2

    x_min = np.min(polygon_points.T[x_idx])
    y_min = np.min(polygon_points.T[y_idx])

    n_patches = patch_nums[x_idx]*patch_nums[y_idx]
    patches_points = np.empty((n_patches, 4, 3))
    i = 0
    for i_x in range(patch_nums[x_idx]):
        for i_y in range(patch_nums[y_idx]):
            points = polygon_points.copy()
            points[0, x_idx] = x_min + i_x * real_size[x_idx]
            points[0, y_idx] = y_min + i_y * real_size[y_idx]
            points[1, x_idx] = x_min + (i_x+1) * real_size[x_idx]
            points[1, y_idx] = y_min + i_y * real_size[y_idx]
            points[3, x_idx] = x_min + i_x * real_size[x_idx]
            points[3, y_idx] = y_min + (i_y+1) * real_size[y_idx]
            points[2, x_idx] = x_min + (i_x+1) * real_size[x_idx]
            points[2, y_idx] = y_min + (i_y+1) * real_size[y_idx]
            patches_points[i] = points
            i += 1

    return patches_points


@numba.jit(nopython=True)
def _calculate_center(points):
    return np.sum(points, axis=-2) / points.shape[-2]

@numba.jit(nopython=True)
def _calculate_size(points):
    vec1 = points[..., 0, :]-points[..., 1, :]
    vec2 = points[..., 1, :]-points[..., 2, :]
    return np.abs(vec1-vec2)

@numba.jit(nopython=True)
def _calculate_area(points):
    vec1 = points[..., 0, :]-points[..., 1, :]
    vec2 = points[..., 1, :]-points[..., 2, :]
    size = vec1-vec2
    return np.abs(
        size[..., 0]*size[..., 1] + size[..., 1]*size[..., 2] \
            + size[..., 0]*size[..., 2])

# @numba.jit(nopython=True)
def _calculate_energy_exchange(
        form_factors_tilde, patches_center, max_order_k, n_patches, n_bins):
    energy_exchange = np.zeros(
        (max_order_k+1, n_patches, n_patches, n_bins+1))
    form_factors_tilde = form_factors_tilde
    patches_center = patches_center

    for ii in range(n_patches**3 * (max_order_k)):
        k = (int(ii/(n_patches**3)) % n_patches)+max_order_k-1
        h = int(ii/(n_patches**2)) % n_patches
        j = int(ii/n_patches) % n_patches
        i = ii % n_patches
        distance = np.linalg.norm(
            patches_center[i]-patches_center[j])
        if k == 2:
            energy_exchange[k, i, j, :-1] += form_factors_tilde[h, i, j, :]
            energy_exchange[k, i, j, -1] = distance
        elif k>2:
            energy_exchange[k, i, j, :-1] += energy_exchange[
                k-1, i, j, :-1]*form_factors_tilde[h, i, j, :]
            energy_exchange[k, i, j, -1] = energy_exchange[
                k-1, i, j, -1] + distance

    return energy_exchange

