"""Module for the radiosity simulation."""
import numpy as np
import pyfar as pf
from . import form_factor, source_energy, receiver_energy, geometry
from . import energy_exchange_recursive as ee_recursive
from . import energy_exchange_order as ee_order


class DRadiosityFast():
    """Radiosity object for directional scattering coefficients."""

    _walls_points: np.ndarray
    _walls_normal: np.ndarray
    _walls_up_vector: np.ndarray
    _patches_points: np.ndarray
    _patches_normal: np.ndarray
    _patch_size: float
    _n_patches: int
    _speed_of_sound: float

    _visibility_matrix: np.ndarray
    _visible_patches: np.ndarray
    _form_factors: np.ndarray
    _form_factors_tilde: np.ndarray

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
    _patch_to_wall_ids: np.ndarray


    def __init__(
            self, walls_points, walls_normal, walls_up_vector,
            patches_points, patches_normal, patch_size, n_patches,
            patch_to_wall_ids):
        """Create a Radiosity object for directional implementation."""
        self._walls_points = walls_points
        self._walls_normal = walls_normal
        self._walls_up_vector = walls_up_vector
        self._patches_points = patches_points
        self._patches_normal = patches_normal
        self._patch_size = patch_size
        self._n_patches = n_patches
        self._patch_to_wall_ids = patch_to_wall_ids
        self._n_bins = None
        self._frequencies = None
        self._sources = None
        self._receivers = None
        self._absorption = None
        self._air_attenuation = None
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
            n_patches, patch_to_wall_ids) = geometry.process_patches(
            walls_points, walls_normal, patch_size, len(polygon_list))
        # create radiosity object
        return cls(
            walls_points, walls_normal, walls_up_vector,
            patches_points, patches_normal, patch_size, n_patches,
            patch_to_wall_ids)

    def bake_geometry(self, ff_method='kang', algorithm='recursive'):
        """Bake the geometry by calculating all the form factors.

        Parameters
        ----------
        ff_method : str, optional
            _description_, by default 'kang'
        algorithm : str, optional
            _description_, by default 'recursive'
            can also be 'order'

        """
        # Check the visibility between patches.
        self._visibility_matrix = geometry.check_visibility(
            self.patches_center, self.patches_normal)

        n_combinations = np.sum(self.visibility_matrix)
        visible_patches = np.empty((n_combinations, 2), dtype=np.int32)
        i_counter = 0
        for i_source in range(self.n_patches):
            for i_receiver in range(self.n_patches):
                if not self.visibility_matrix[i_source, i_receiver]:
                    continue
                visible_patches[i_counter, 0] = i_source
                visible_patches[i_counter, 1] = i_receiver
                i_counter += 1
        self._visible_patches = visible_patches

        if ff_method == 'kang':
            self._form_factors = form_factor.kang(
                self.patches_center, self.patches_normal,
                self.patches_size, self._visible_patches)
        elif ff_method == 'universal':
            self._form_factors = form_factor.universal(
                self.patches_points, self.patches_normal,
                self.patches_area, self._visible_patches)
        else:
            raise NotImplementedError()

        # Calculate the form factors with directivity.
        if self._sources is not None:
            sources_array = np.array([s.cartesian for s in self._sources])
            receivers_array = np.array([s.cartesian for s in self._receivers])
            scattering_index = np.array(self._scattering_index)
            scattering = np.array(self._scattering)
        else:
            sources_array = None
            receivers_array = None
            scattering_index = None
            scattering = None

        if self._absorption is None:
            absorption = None
            absorption_index = None
        else:
            absorption = np.atleast_2d(np.array(self._absorption))
            absorption_index = np.array(self._absorption_index)

        n_bins = 1 if self._n_bins is None else self._n_bins

        if algorithm == 'recursive':
            self._form_factors_tilde = \
                form_factor._form_factors_with_directivity(
                self.visibility_matrix, self.form_factors, n_bins,
                self.patches_center,
                self._air_attenuation, absorption,
                absorption_index,
                self._patch_to_wall_ids, scattering,
                scattering_index,
                sources_array, receivers_array)
        elif algorithm == 'order':
            self._form_factors_tilde = \
                form_factor._form_factors_with_directivity_dim(
                self.visibility_matrix, self.form_factors, n_bins,
                self.patches_center,
                self._air_attenuation, absorption,
                absorption_index,
                self._patch_to_wall_ids, scattering,
                scattering_index,
                sources_array, receivers_array)

    def init_source_energy(
            self, source_position:np.ndarray, ff_method="kang", algorithm='recursive'):
        """Initialize the source energy."""
        source_position = np.array(source_position)
        patch_to_wall_ids = self._patch_to_wall_ids
        absorption = np.atleast_2d(np.array(self._absorption))
        absorption_index = self._absorption_index
        sources = np.array([s.cartesian for s in self._sources])
        receivers = np.array([s.cartesian for s in self._receivers])
        scattering = np.array(self._scattering)
        scattering_index = self._scattering_index
        form_factors = self.form_factors
        patches_center = self.patches_center
        if ff_method == "kang":
            energy_0, distance_0 = source_energy._init_energy_kang(
                source_position, patches_center, self.patches_normal,
                self._air_attenuation, self.patches_size, self.n_bins)
        elif ff_method == "universal":
            energy_0, distance_0 = source_energy._init_energy_universal(
                source_position, patches_center, self.patches_points,
                self._air_attenuation, self.n_bins)
        else:
            raise NotImplementedError()

        if algorithm == 'recursive':
            energy_1, distance_1 = ee_recursive._init_energy_1(
                energy_0, distance_0, source_position,
                patches_center, self._visible_patches,
                self._air_attenuation, self._n_bins, patch_to_wall_ids,
                absorption, absorption_index,
                form_factors, sources, receivers,
                scattering, scattering_index)
            self.energy_0 = energy_0
            self.distance_0 = distance_0
            self.energy_1 = energy_1
            self.distance_1 = distance_1
        elif algorithm == 'order':
            energy_0_dir = ee_order._add_directional(
                energy_0, source_position,
                patches_center, self._n_bins, patch_to_wall_ids,
                absorption, absorption_index,
                sources, receivers,
                scattering, scattering_index)
            self.energy_0_dir = energy_0_dir
            self.distance_0 = distance_0
        else:
            raise NotImplementedError()

    def calculate_energy_exchange(
            self, receiver_pos, speed_of_sound,
            histogram_time_resolution,
            histogram_length, ff_method='kang', algorithm='recursive',
            threshold=1e-6, max_time=np.inf, max_depth=-1, recalculate=False):
        """Calculate the energy exchange."""
        n_samples = int(histogram_length/histogram_time_resolution)
        receiver_pos = np.array(receiver_pos)
        if receiver_pos.ndim==1:
            receiver_pos=receiver_pos[np.newaxis,:]
        ir = np.array([[np.zeros((n_samples)) for _ in range(self.n_bins)]
                                for _ in range(receiver_pos.shape[0])])
        patches_center = self.patches_center
        patch_receiver_distance = np.empty([receiver_pos.shape[0],
                                            self.n_patches,patches_center.shape[-1]])
        for i in range(receiver_pos.shape[0]):
            patch_receiver_distance[i] = patches_center - receiver_pos[i]
        air_attenuation = self._air_attenuation
        patches_normal = self._patches_normal
        patches_points = self._patches_points
        distance_0 = self.distance_0
        n_patches = self.n_patches
        n_bins = self.n_bins
        distance_i_j = np.empty((n_patches, n_patches))
        for i in range(n_patches):
            for j in range(n_patches):
                distance_i_j[i, j] = np.linalg.norm(
                    patches_center[i, :]-patches_center[j, :])
        if ff_method == 'kang':
            patch_receiver_energy = receiver_energy._kang(
                patch_receiver_distance, patches_normal, air_attenuation)
        elif ff_method == 'universal':
            pass
        else:
            raise NotImplementedError()
        if algorithm == 'recursive':
            # add first 2 order energy exchange
            ir = ir.T
            energy_0 = self.energy_0
            energy_1 = self.energy_1
            distance_1 = self.distance_1
            ee_recursive._calculate_energy_exchange_second_order(
                ir, energy_0, distance_0, energy_1, distance_1,
                patch_receiver_distance, patch_receiver_energy ,speed_of_sound,
                histogram_time_resolution, n_patches, n_bins)
            # add remaining energy
            ee_recursive._calculate_energy_exchange_recursive(
                ir, energy_1, distance_1, distance_i_j,
                self._form_factors_tilde,
                self.n_patches, patch_receiver_distance, patch_receiver_energy,
                speed_of_sound, histogram_time_resolution,
                threshold=threshold, max_time=max_time, max_depth=max_depth)
            return ir.T
        elif algorithm == 'order':
            energy_0_dir = self.energy_0_dir
            # assert max_depth>=1, "max_depth must be larger than 1"
            if not hasattr(self, 'E_matrix_total') or recalculate:
                self.E_matrix_total = ee_order.energy_exchange(
                    n_samples, energy_0_dir, distance_0, distance_i_j,
                    self._form_factors_tilde,
                    speed_of_sound, histogram_time_resolution, max_depth,
                    self._visible_patches)
        else:
            raise NotImplementedError()
        
    def collect_receiver_energy(self, receiver_pos,
            speed_of_sound, histogram_time_resolution, propagation_fx=False):

        air_attenuation = self._air_attenuation
        patches_points = self._patches_points
        n_patches = self.n_patches
        n_bins = self.n_bins

        if receiver_pos.ndim==1:
            receiver_pos=receiver_pos[np.newaxis,:]

        n_receivers = receiver_pos.shape[0]

        patches_center = self.patches_center
        patch_receiver_distance = np.empty([n_receivers,
                                            self.n_patches,patches_center.shape[-1]])
        
        E_matrix = np.zeros((n_receivers, n_patches, n_bins, self.E_matrix_total.shape[-1]))
        histogram_out = np.zeros((n_receivers, n_patches, n_bins, self.E_matrix_total.shape[-1]))

        for i in range(n_receivers):
            patch_receiver_distance[i] = patches_center - receiver_pos[i]
        
            # geometrical weighting
            patch_receiver_energy = receiver_energy._universal(
                    receiver_pos[i], patches_points)
            
            # access histograms with correct scattering weighting
            receivers_array = np.array([s.cartesian for s in self._receivers])
                # scattering_index = np.array(self._scattering_index)
            receiver_idx = geometry.get_scattering_data_receiver_index(
                patches_center, receiver_pos[i], receivers_array,
                self._patch_to_wall_ids)
        
            assert receiver_idx.shape[0] == self.n_patches
            assert len(receiver_idx.shape) == 1

            for k in range(n_patches):
                E_matrix[i,k,:]= self.E_matrix_total[k,receiver_idx[k],:] * patch_receiver_energy[k]

            if propagation_fx:
                # accumulate the patch energies towards the receiver 
                # with atmospheric effects (delay, atmospheric attenuation)
                histogram_out[i] = ee_order._collect_receiver_energy(
                        E_matrix[i],
                        np.linalg.norm(patch_receiver_distance[i], axis=1), speed_of_sound,
                        histogram_time_resolution, air_attenuation=air_attenuation)
            else:
                histogram_out = E_matrix
                
        return histogram_out

    def set_wall_absorption(self, wall_indexes, absorption:pf.FrequencyData):
        """Set the wall absorption.

        Parameters
        ----------
        wall_indexes : list[int]
            list of walls for the scattering data
        absorption : pf.FrequencyData
            absorption coefficient of cshape (1, )

        """
        self._check_set_frequency(absorption.frequencies)
        if self._absorption is None:
            self._absorption_index = np.empty((self.n_walls), dtype=np.int64)
            self._absorption_index.fill(-1)
            self._absorption = []

        self._absorption.append(np.atleast_1d(absorption.freq.squeeze()))
        self._absorption_index[wall_indexes] = len(self._absorption)-1


    def set_air_attenuation(self, air_attenuation:pf.FrequencyData):
        """Set air attenuation factor in Np/m.

        Parameters
        ----------
        air_attenuation : pf.FrequencyData
            Air attenuation factor in Np/m.

        """
        self._check_set_frequency(air_attenuation.frequencies)
        self._air_attenuation = np.atleast_1d(air_attenuation.freq.squeeze())

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
            sources_rot, receivers_rot = _rotate_coords_to_normal(
                self.walls_normal[i], self.walls_up_vector[i],
                sources, receivers)
            self._sources[i] = sources_rot
            self._receivers[i] = receivers_rot

        self._scattering.append(scattering.freq)
        self._scattering_index[wall_indexes] = len(self._scattering)-1

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
        return geometry._calculate_area(self._walls_points)

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
        return geometry._calculate_center(self._walls_points)

    @property
    def walls_up_vector(self):
        """Return the up vector of the walls."""
        return self._walls_up_vector

    @property
    def patches_area(self):
        """Return the area of the patches."""
        return geometry._calculate_area(self._patches_points)

    @property
    def patches_center(self):
        """Return the center of the patches."""
        return geometry._calculate_center(self._patches_points)

    @property
    def patches_size(self):
        """Return the size of the patches."""
        return geometry._calculate_size(self._patches_points)

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
    def speed_of_sound(self):
        """Return the speed of sound in m/s."""
        return self._speed_of_sound



def _rotate_coords_to_normal(
        wall_normal:np.ndarray, wall_up_vector:np.ndarray,
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
