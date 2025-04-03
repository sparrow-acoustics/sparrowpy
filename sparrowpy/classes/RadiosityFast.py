"""Module for the radiosity simulation."""
import numpy as np
import pyfar as pf
from sparrowpy.radiosity_fast import (
    form_factor, source_energy, receiver_energy, geometry)
from sparrowpy.radiosity_fast import energy_exchange_order as ee_order


class DirectionalRadiosityFast():
    """Radiosity object for directional scattering coefficients."""

    _walls_points: np.ndarray
    _walls_normal: np.ndarray
    _walls_up_vector: np.ndarray
    _patches_points: np.ndarray
    _patches_normal: np.ndarray
    _n_patches: int
    _patch_to_wall_ids: np.ndarray

    # geometrical data
    _visibility_matrix: np.ndarray
    _visible_patches: np.ndarray
    _form_factors: np.ndarray
    _form_factors_tilde: np.ndarray

    # general data for material and medium data
    _n_bins: int
    _frequencies: np.ndarray
    _brdf: np.ndarray
    _brdf_index: np.ndarray
    _brdf_incoming_directions: list[pf.Coordinates]
    _brdf_outgoing_directions: list[pf.Coordinates]

    _air_attenuation: np.ndarray
    _speed_of_sound: float

    # etc metadata
    _etc_time_resolution: float
    _etc_duration: float

    # etc results
    _distance_patches_to_source: np.ndarray
    _energy_init_etc: np.ndarray
    _energy_exchange_etc: np.ndarray


    def __init__(
            self, walls_points, walls_normal, walls_up_vector,
            patches_points, patches_normal, n_patches,
            patch_to_wall_ids):
        """Create a Radiosity object for directional implementation."""
        self._walls_points = walls_points
        self._walls_normal = walls_normal
        self._walls_up_vector = walls_up_vector
        self._patches_points = patches_points
        self._patches_normal = patches_normal
        self._n_patches = n_patches
        self._patch_to_wall_ids = patch_to_wall_ids

        # geometrical data
        self._visibility_matrix = None
        self._visible_patches = None
        self._form_factors = None
        self._form_factors_tilde = None

        # general data for material and medium data
        self._n_bins = None
        self._frequencies = None
        self._brdf = None
        self._brdf_index = None
        self._brdf_incoming_directions = None
        self._brdf_outgoing_directions = None

        self._air_attenuation = None
        self._speed_of_sound = None

        # etc metadata
        self._etc_time_resolution = None
        self._etc_duration = None

        # etc results
        self._distance_patches_to_source = None
        self._energy_init_etc = None
        self._energy_exchange_etc = None

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
            patches_points, patches_normal, n_patches,
            patch_to_wall_ids)

    def bake_geometry(self):
        """Bake the geometry by calculating all the form factors.

        """
        # Check the visibility between patches.
        self._visibility_matrix = geometry.check_visibility(
            self.patches_center, self.patches_normal, self.patches_points)

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

        self._form_factors = form_factor.universal(
            self.patches_points, self.patches_normal,
            self.patches_area, self._visible_patches)

        # Calculate the form factors with directivity.
        if self._brdf_incoming_directions is not None:
            sources_array = np.array(
                [s.cartesian for s in self._brdf_incoming_directions])
            receivers_array = np.array(
                [s.cartesian for s in self._brdf_outgoing_directions])
            scattering_index = np.array(self._brdf_index)
            scattering = np.array(self._brdf)
        else:
            sources_array = None
            receivers_array = None
            scattering_index = None
            scattering = None

        n_bins = 1 if self._n_bins is None else self._n_bins

        self._form_factors_tilde = \
            form_factor._form_factors_with_directivity_dim(
            self.visibility_matrix, self.form_factors, n_bins,
            self.patches_center, self.patches_area,
            self._air_attenuation,
            self._patch_to_wall_ids, scattering,
            scattering_index,
            sources_array, receivers_array)

    def init_source_energy(
            self, source:pf.Coordinates):
        """Initialize the source energy."""
        source_position = source.cartesian
        patch_to_wall_ids = self._patch_to_wall_ids
        sources = np.array(
            [s.cartesian for s in self._brdf_incoming_directions])
        receivers = np.array(
            [s.cartesian for s in self._brdf_outgoing_directions])
        scattering = np.array(self._brdf)
        scattering_index = self._brdf_index
        patches_center = self.patches_center
        energy_0, distance_0 = source_energy._init_energy_universal(
            source_position, patches_center, self.patches_points,
            self._air_attenuation, self.n_bins)
        energy_0_dir = ee_order._add_directional(
            energy_0, source_position,
            patches_center, self._n_bins, patch_to_wall_ids,
            sources, receivers,
            scattering, scattering_index)
        self._energy_init_etc = energy_0_dir
        self._distance_patches_to_source = distance_0

    def calculate_energy_exchange(
            self, speed_of_sound,
            etc_time_resolution,
            etc_duration,
            max_reflection_order=-1,
            recalculate=False):
        """Calculate the energy exchange between patches.
        # todo? make first order to first reflection.
        """
        n_samples = int(etc_duration/etc_time_resolution)

        patches_center = self.patches_center
        distance_0 = self._distance_patches_to_source
        n_patches = self.n_patches
        distance_i_j = np.empty((n_patches, n_patches))

        for i in range(n_patches):
            for j in range(n_patches):
                distance_i_j[i, j] = np.linalg.norm(
                    patches_center[i, :]-patches_center[j, :])

        energy_0_dir = self._energy_init_etc

        if self._energy_exchange_etc is None or recalculate:
            if max_reflection_order < 1:
                self._energy_exchange_etc = \
                    ee_order.energy_exchange_init_energy(
                        n_samples, energy_0_dir, distance_0,
                        speed_of_sound, etc_time_resolution,
                        )
            else:
                self._energy_exchange_etc = ee_order.energy_exchange(
                    n_samples, energy_0_dir, distance_0, distance_i_j,
                    self._form_factors_tilde,
                    speed_of_sound, etc_time_resolution,
                    max_reflection_order,
                    self._visible_patches)

        self._etc_time_resolution = etc_time_resolution
        self._speed_of_sound = speed_of_sound
        self._etc_duration = etc_duration

    def collect_energy_receiver_mono(self, receivers):
        """Collect the energy at the receivers.

        Parameters
        ----------
        receivers : pf.Coordinates
            receiver Coordinates in of cshape (n_receivers).

        Returns
        -------
        etc : pf.TimeData
            energy collected at the receiver in cshape
            (n_receivers, n_bins)
        """
        etc = self.collect_energy_receiver_patchwise(receivers)
        etc.time = np.sum(etc.time, axis=1)
        return etc

    def collect_energy_receiver_patchwise(self, receivers):
        """Collect the energy for each patch at the receivers without summing
        up the patches.

        Parameters
        ----------
        receivers : pf.Coordinates
            receiver Coordinates in of cshape (n_receivers).

        Returns
        -------
        etc : pf.TimeData
            energy collected at the receiver in cshape
            (n_receivers, n_patches, n_bins)
        """
        if not isinstance(receivers, pf.Coordinates):
            raise ValueError(
                "Receiver positions must be of type pf.Coordinates")
        if receivers.cdim != 1:
            raise ValueError(
                "Receiver positions must be of shape (n_receivers, 3)")
        etc_data = self._collect_energy_patches(
            receivers.cartesian, propagation_fx=True)
        times = np.arange(etc_data.shape[-1]) * self._etc_time_resolution
        return pf.TimeData(etc_data, times)

    def _collect_energy_patches(
            self, receiver_pos,
            propagation_fx=False):
        """Collect patch histograms as detected by receiver."""
        air_attenuation = self._air_attenuation
        patches_points = self._patches_points
        n_patches = self.n_patches
        n_bins = self.n_bins

        receiver_pos = np.atleast_2d(receiver_pos)

        n_receivers = receiver_pos.shape[0]

        patches_center = self.patches_center
        patches_receiver_distance = np.empty(
            [n_receivers, self.n_patches,patches_center.shape[-1]])

        E_matrix = np.empty(
            (n_patches, n_bins, self._energy_exchange_etc.shape[-1]))
        histogram_out = np.empty((
            n_receivers, n_patches, n_bins,
            self._energy_exchange_etc.shape[-1]))

        for i in range(n_receivers):
            patches_receiver_distance = patches_center - receiver_pos[i]

            # geometrical weighting
            patch_receiver_energy = receiver_energy._universal(
                    receiver_pos[i], patches_points)

            # access histograms with correct scattering weighting
            receivers_array = np.array(
                [s.cartesian for s in self._brdf_outgoing_directions])

            receiver_idx = geometry.get_scattering_data_receiver_index(
                patches_center, receiver_pos[i], receivers_array,
                self._patch_to_wall_ids)

            assert receiver_idx.shape[0] == self.n_patches
            assert len(receiver_idx.shape) == 1

            for k in range(n_patches):
                E_matrix[k,:]= (self._energy_exchange_etc[k,receiver_idx[k],:]
                                                    * patch_receiver_energy[k])

            if propagation_fx:
                # accumulate the patch energies towards the receiver
                # with atmospheric effects (delay, atmospheric attenuation)
                histogram_out[i] = ee_order._collect_receiver_energy(
                        E_matrix,
                        np.linalg.norm(patches_receiver_distance, axis=1),
                                    self.speed_of_sound,
                                    self._etc_time_resolution,
                                    air_attenuation=air_attenuation)
            else:
                histogram_out[i] = E_matrix

        return histogram_out

    def set_air_attenuation(self, air_attenuation:pf.FrequencyData):
        """Set air attenuation factor in Np/m.

        Parameters
        ----------
        air_attenuation : pf.FrequencyData
            Air attenuation factor in Np/m.

        """
        self._check_set_frequency(air_attenuation.frequencies)
        self._air_attenuation = np.atleast_1d(air_attenuation.freq.squeeze())

    def set_wall_brdf(
            self,
            wall_indexes:list[int],
            brdf:pf.FrequencyData,
            incoming_directions:pf.Coordinates,
            outgoing_directions:pf.Coordinates):
        """Set the wall scattering.

        Parameters
        ----------
        wall_indexes : list[int]
            list of walls for the scattering data
        brdf : pf.FrequencyData
            brdf data of cshape (n_sources, n_receivers)
        incoming_directions : pf.Coordinates
            source coordinates
        outgoing_directions : pf.Coordinates
            receiver coordinates

        """
        assert (incoming_directions.z >= 0).all(), \
            "Sources must be in the positive half space"
        assert (outgoing_directions.z >= 0).all(), \
            "Receivers must be in the positive half space"
        self._check_set_frequency(brdf.frequencies)
        if self._brdf_incoming_directions is None:
            self._brdf_incoming_directions = np.empty(
                (self.n_walls), dtype=pf.Coordinates)
            self._brdf_outgoing_directions = np.empty(
                (self.n_walls), dtype=pf.Coordinates)
            self._brdf_index = np.empty((self.n_walls), dtype=np.int64)
            self._brdf_index.fill(-1)
            self._brdf = []

        for i in wall_indexes:
            incoming_rot, outgoing_rot = _rotate_coords_to_normal(
                self.walls_normal[i], self.walls_up_vector[i],
                incoming_directions, outgoing_directions)
            self._brdf_incoming_directions[i] = incoming_rot
            self._brdf_outgoing_directions[i] = outgoing_rot

        self._brdf.append(brdf.freq*np.pi)
        self._brdf_index[wall_indexes] = len(self._brdf)-1

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
