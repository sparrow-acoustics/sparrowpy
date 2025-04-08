"""Module for the radiosity simulation."""
import numpy as np
import pyfar as pf
import sparrowpy.form_factor.universal as form_factor
from sparrowpy import ( geometry )
from sparrowpy.utils import blender
try:
    import numba
    prange = numba.prange
except ImportError:
    numba = None
    prange = range

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
            n_patches, patch_to_wall_ids) = geometry._process_patches(
            walls_points, walls_normal, patch_size, len(polygon_list))
        # create radiosity object
        return cls(
            walls_points, walls_normal, walls_up_vector,
            patches_points, patches_normal, n_patches,
            patch_to_wall_ids)

    @classmethod
    def from_file(cls, filename: str, patch_size=1.0,
                       auto_walls=True, auto_patches=True):
        """Create a Radiosity object ffrom a blender file.

        """
        geom_data = blender.read_geometry_file(filename,
                                           auto_walls=auto_walls,
                                           patches_from_model=auto_patches)

        walls   = geom_data["wall"]

        ## save wall information
        walls_normal = walls["normal"]
        walls_up_vector = np.empty_like(walls["up"])

        walls_points=np.empty((len(walls["conn"]),
                               len(walls["conn"][0]),
                               walls["verts"].shape[-1]))

        for wallID in range(len(walls_normal)):
            walls_points[wallID] = walls["verts"][walls["conn"][wallID]]
            walls_up_vector[wallID] = walls["up"][wallID]

        if bool(geom_data["patch"]):
            patches = geom_data["patch"]

            ## save patch information
            n_patches = len(patches["map"])

            patch_to_wall_ids = patches["map"]

            patches_normal = walls["normal"][patch_to_wall_ids]

            patches_points = np.empty((n_patches,
                                len(patches["conn"][0]),
                                patches["verts"].shape[-1]))

            for patchID in range(n_patches):
                patches_points[patchID] = patches["verts"][
                                                patches["conn"][patchID]
                                                ]

        else:
            (
            patches_points, patches_normal,
            n_patches, patch_to_wall_ids) = geometry._process_patches(
            walls_points, walls_normal, patch_size, len(walls_normal))

        # create radiosity object
        return cls(
            walls_points, walls_normal, walls_up_vector,
            patches_points, patches_normal, n_patches,
            patch_to_wall_ids)

    def bake_geometry(self):
        """Bake the geometry by calculating all the form factors.

        """
        # Check the visibility between patches.
        self._visibility_matrix = geometry._check_visibility(
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

        self._form_factors = form_factor.patch2patch_ff_universal(
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
            _form_factors_with_directivity_dim(
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
        energy_0, distance_0 = form_factor._source2patch_energy_universal(
            source_position, patches_center, self.patches_points,
            self._air_attenuation, self.n_bins)
        energy_0_dir = _add_directional(
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
                    _energy_exchange_init_energy(
                        n_samples, energy_0_dir, distance_0,
                        speed_of_sound, etc_time_resolution,
                        )
            else:
                self._energy_exchange_etc = _energy_exchange(
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
            patch_receiver_energy=form_factor._patch2receiver_energy_universal(
                    receiver_pos[i], patches_points)

            # access histograms with correct scattering weighting
            receivers_array = np.array(
                [s.cartesian for s in self._brdf_outgoing_directions])

            receiver_idx = get_scattering_data_receiver_index(
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
                histogram_out[i] = _collect_receiver_energy(
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
        """Set the wall BRDF representing scattering and absorption.

        For the incoming and outgoing directions, the radius is ignored
        as it only represents a direction. The BRDF assumes an up vector
        of (1, 0, 0) and a normal vector of (0, 0, 1). Therefore, the
        incoming and outgoing directions must not have negative z-components.
        For each wall, the incoming and outgoing directions are rotated
        to align with the given wall's normal vector and up vector.

        Parameters
        ----------
        wall_indexes : list[int]
            List of wall indices for the given BRDF data.
        brdf : pf.FrequencyData
            BRDF data with shape
            (n_incoming_directions, n_outgoing_directions).
        incoming_directions : pf.Coordinates
            Incoming directions of the BRDF data.
        outgoing_directions : pf.Coordinates
            Outgoing directions of the BRDF data.

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

def _add_directional(
        energy_0, source_position: np.ndarray,
        patches_center: np.ndarray, n_bins:float, patch_to_wall_ids:np.ndarray,
        sources: np.ndarray, receivers: np.ndarray,
        scattering: np.ndarray, scattering_index: np.ndarray):
    """Add scattering and absorption to the initial energy from the source.

    Parameters
    ----------
    energy_0 : np.ndarray
        energy of all patches of shape (n_patches, n_bins)
    source_position : np.ndarray
        source position of shape (3,)
    patches_center : np.ndarray
        center of all patches of shape (n_patches, 3)
    n_bins : float
        number of frequency bins.
    patch_to_wall_ids : np.ndarray
        indexes from each patch to the wall of shape (n_patches)
    sources : np.ndarray
        source positions of shape (n_walls, n_sources, 3)
    receivers : np.ndarray
        receiver positions of shape (n_walls, n_receivers, 3)
    scattering : np.ndarray
        scattering data of shape (n_walls, n_sources, n_receivers, n_bins)
    scattering_index : np.ndarray
        mapping from the wall id to scattering database index (n_walls)

    Returns
    -------
    energy : np.ndarray
        energy of all patches of shape (n_patches, n_directions, n_bins)

    """
    n_patches = patches_center.shape[0]
    n_directions = receivers.shape[1]
    energy_0_directivity = np.zeros((n_patches, n_directions, n_bins))
    for i in prange(n_patches):
        wall_id_i = int(patch_to_wall_ids[i])
        scattering_factor = get_scattering_data_source(
            source_position, patches_center[i],
            sources, wall_id_i, scattering, scattering_index)

        energy_0_directivity[i, :, :] = energy_0[i] \
            * scattering_factor

    return energy_0_directivity


def _energy_exchange_init_energy(
        n_samples, energy_0_directivity, distance_0,
        speed_of_sound, histogram_time_resolution):
    """Calculate energy exchange between patches.

    Parameters
    ----------
    n_samples : int
        number of samples of the histogram.
    energy_0_directivity : np.ndarray
        init energy of all patches of shape (n_patches, n_directions, n_bins)
    distance_0 : np.ndarray
        distance from the source to all patches of shape (n_patches)
    speed_of_sound : float
        speed of sound in m/s.
    histogram_time_resolution : float
        time resolution of the histogram in s.

    Returns
    -------
    E_matrix_total : np.ndarray
        energy of all patches of shape
        (n_patches, n_directions, n_bins, n_samples)

    """
    n_patches = energy_0_directivity.shape[0]
    n_directions = energy_0_directivity.shape[1]
    n_bins = energy_0_directivity.shape[2]
    E_matrix_total = np.zeros((n_patches, n_directions, n_bins, n_samples))
    for i in prange(n_patches):
        n_delay_samples = int(
            distance_0[i]/speed_of_sound/histogram_time_resolution)
        E_matrix_total[i, :, :, n_delay_samples] += energy_0_directivity[i]
    return E_matrix_total


def _energy_exchange(
        n_samples, energy_0_directivity, distance_0, distance_ij,
        form_factors_tilde,
        speed_of_sound, histogram_time_resolution, max_order, visible_patches):
    """Calculate energy exchange between patches.

    Parameters
    ----------
    n_samples : int
        number of samples of the histogram.
    energy_0_directivity : np.ndarray
        init energy of all patches of shape (n_patches, n_directions, n_bins)
    distance_0 : np.ndarray
        distance from the source to all patches of shape (n_patches)
    distance_ij : np.ndarray
        distance between all patches of shape (n_patches, n_patches)
    form_factors_tilde : np.ndarray
        form factors of shape (n_patches, n_patches, n_directions, n_bins)
    speed_of_sound : float
        speed of sound in m/s.
    histogram_time_resolution : float
        time resolution of the histogram in s.
    max_order : int
        maximum order of reflections.
    visible_patches : np.ndarray
        indexes of all visible patches of shape (n_visible, 2)

    Returns
    -------
    E_matrix_total : np.ndarray
        energy of all patches of shape
        (n_patches, n_directions, n_bins, n_samples)

    """
    n_patches = form_factors_tilde.shape[0]
    n_directions = form_factors_tilde.shape[2]
    n_bins = energy_0_directivity.shape[2]
    form_factors_tilde = form_factors_tilde[..., np.newaxis]
    E_matrix_total  = _energy_exchange_init_energy(
        n_samples, energy_0_directivity, distance_0, speed_of_sound,
        histogram_time_resolution)
    E_matrix = np.zeros((2, n_patches, n_directions, n_bins, n_samples))
    E_matrix[0] += E_matrix_total
    if max_order == 0:
        return E_matrix_total
    for k in range(max_order):
        current_index = (1+k) % 2
        E_matrix[current_index, :, :, :] = 0
        for ii in range(visible_patches.shape[0]):
            for jj in range(2):
                if jj == 0:
                    i = visible_patches[ii, 0]
                    j = visible_patches[ii, 1]
                else:
                    j = visible_patches[ii, 0]
                    i = visible_patches[ii, 1]
                n_delay_samples = int(
                    distance_ij[i, j]/speed_of_sound/histogram_time_resolution)
                if n_delay_samples > 0:
                    E_matrix[current_index, j, :, :, n_delay_samples:] += \
                        form_factors_tilde[i, j] * E_matrix[
                            current_index-1, i, :, :, :-n_delay_samples]
                else:
                    E_matrix[current_index, j, :, :, :] += form_factors_tilde[
                        i, j] * E_matrix[current_index-1, i, :, :, :]
        E_matrix_total += E_matrix[current_index]
    return E_matrix_total


def _collect_receiver_energy(
        E_matrix_total, patch_receiver_distance,
        speed_of_sound, histogram_time_resolution, air_attenuation):
    """Collect the energy at the receiver.

    Parameters
    ----------
    E_matrix_total : np.ndarray
        energy of all patches of shape
        (n_patches, n_directions, n_bins, n_samples)
    patch_receiver_distance : np.ndarray
        distance from the patch to the receiver of shape (n_patches)
    speed_of_sound : float
        speed of sound in m/s.
    histogram_time_resolution : float
        time resolution of the histogram in s.
    air_attenuation : np.ndarray
        air attenuation factor for each frequency bin of shape (n_bins)

    Returns
    -------
    ir : np.ndarray
        impulse response of shape (n_samples, n_bins)

    """
    E_mat_out = np.zeros_like(E_matrix_total)
    n_patches = E_matrix_total.shape[0]
    n_bins = E_matrix_total.shape[1]

    for i in prange(n_patches):
        n_delay_samples = int(np.ceil(
            patch_receiver_distance[i]/speed_of_sound/histogram_time_resolution))
        for j in range(n_bins):
            E_mat_out[i,j] = np.roll(
                E_matrix_total[i,j]*np.exp(-air_attenuation[j]*patch_receiver_distance[i]),
                n_delay_samples)

    return E_mat_out

def _form_factors_with_directivity(
        visibility_matrix, form_factors, n_bins, patches_center,
        patches_area, air_attenuation,
        absorption, absorption_index, patch_to_wall_ids,
        scattering, scattering_index, sources, receivers):
    """Calculate the form factors with directivity."""
    n_patches = patches_center.shape[0]
    form_factors_tilde = np.zeros((n_patches, n_patches, n_patches, n_bins))
    # loop over previous patches, current and next patch

    for ii in prange(n_patches**3):
        h = ii % n_patches
        i = int(ii/n_patches) % n_patches
        j = int(ii/n_patches**2) % n_patches
        visible_hi = visibility_matrix[
            h, i] if h < i else visibility_matrix[i, h]
        visible_ij = visibility_matrix[
            i, j] if i < j else visibility_matrix[j, i]
        if visible_hi and visible_ij:
            difference_receiver = patches_center[i]-patches_center[j]
            wall_id_i = int(patch_to_wall_ids[i])
            difference_receiver /= np.linalg.norm(difference_receiver)
            ff = form_factors[i, j] if i<j else (form_factors[j, i]
                                                 *patches_area[i]/patches_area[j])

            distance = np.linalg.norm(difference_receiver)
            form_factors_tilde[h, i, j, :] = ff
            if air_attenuation is not None:
                form_factors_tilde[h, i, j, :] = form_factors_tilde[
                    h, i, j, :] * np.exp(-air_attenuation * distance)

            if scattering is not None:
                scattering_factor = get_scattering_data(
                    patches_center[h], patches_center[i], patches_center[j],
                    sources, receivers, wall_id_i,
                    scattering, scattering_index)
                form_factors_tilde[h, i, j, :] = form_factors_tilde[
                    h, i, j, :] * scattering_factor

            if absorption is not None:
                source_wall_idx = absorption_index[wall_id_i]
                form_factors_tilde[h, i, j, :] = form_factors_tilde[
                    h, i, j, :] * (1-absorption[source_wall_idx])

    return form_factors_tilde


def _form_factors_with_directivity_dim(
        visibility_matrix, form_factors, n_bins, patches_center,
        patches_area,
        air_attenuation,
        patch_to_wall_ids,
        scattering, scattering_index, sources, receivers):
    """Calculate the form factors with directivity."""
    n_patches = patches_center.shape[0]
    n_directions = receivers.shape[1] if receivers is not None else 1
    form_factors_tilde = np.zeros((n_patches, n_patches, n_directions, n_bins))
    # loop over previous patches, current and next patch

    for ii in prange(n_patches**2):
        i = ii % n_patches
        j = int(ii/n_patches) % n_patches
        visible_ij = visibility_matrix[
            i, j] if i < j else visibility_matrix[j, i]
        if visible_ij:
            difference_receiver = patches_center[i]-patches_center[j]
            wall_id_i = int(patch_to_wall_ids[i])
            difference_receiver /= np.linalg.norm(difference_receiver)
            ff = form_factors[i, j] if i<j else (form_factors[j, i]
                                                 *patches_area[j]/patches_area[i])

            distance = np.linalg.norm(difference_receiver)
            form_factors_tilde[i, j, :, :] = ff
            if air_attenuation is not None:
                form_factors_tilde[i, j, :, :] = form_factors_tilde[
                    i, j, :, :] * np.exp(-air_attenuation * distance)

            if scattering is not None:
                scattering_factor = get_scattering_data_source(
                    patches_center[i], patches_center[j],
                    sources, wall_id_i,
                    scattering, scattering_index)
                form_factors_tilde[i, j, :, :] = form_factors_tilde[
                    i, j, :, :] * scattering_factor

    return form_factors_tilde


## scattering data access

def get_scattering_data_receiver_index(
        pos_i:np.ndarray, pos_j:np.ndarray,
        receivers:np.ndarray, wall_id_i:np.ndarray,
        ):
    """Get scattering data depending on previous, current and next position.

    Parameters
    ----------
    pos_i : np.ndarray
        current position of shape (3)
    pos_j : np.ndarray
        next position of shape (3)
    receivers : np.ndarray
        receiver directions of all walls of shape (n_walls, n_receivers, 3)
    wall_id_i : np.ndarray
        current wall id to get write directional data

    Returns
    -------
    scattering_factor: float
        scattering factor from directivity

    """
    n_patches = pos_i.shape[0] if pos_i.ndim > 1 else 1
    receiver_idx = np.empty((n_patches), dtype=np.int64)

    for i in range(n_patches):
        difference_receiver = pos_i[i]-pos_j
        difference_receiver /= np.linalg.norm(
            difference_receiver)
        receiver_idx[i] = np.argmin(np.sum(
            (receivers[wall_id_i[i], :]-difference_receiver)**2, axis=-1),
            axis=-1)


    return receiver_idx


def get_scattering_data(
        pos_h:np.ndarray, pos_i:np.ndarray, pos_j:np.ndarray,
        sources:np.ndarray, receivers:np.ndarray, wall_id_i:np.ndarray,
        scattering:np.ndarray, scattering_index:np.ndarray):
    """Get scattering data depending on previous, current and next position.

    Parameters
    ----------
    pos_h : np.ndarray
        previous position of shape (3)
    pos_i : np.ndarray
        current position of shape (3)
    pos_j : np.ndarray
        next position of shape (3)
    sources : np.ndarray
        source directions of all walls of shape (n_walls, n_sources, 3)
    receivers : np.ndarray
        receiver directions of all walls of shape (n_walls, n_receivers, 3)
    wall_id_i : np.ndarray
        current wall id to get write directional data
    scattering : np.ndarray
        scattering data of shape (n_scattering, n_sources, n_receivers, n_bins)
    scattering_index : np.ndarray
        index of the scattering data of shape (n_walls)

    Returns
    -------
    scattering_factor: float
        scattering factor from directivity

    """
    difference_source = pos_h-pos_i
    difference_receiver = pos_i-pos_j

    difference_source /= np.linalg.norm(difference_source)
    difference_receiver /= np.linalg.norm(difference_receiver)
    source_idx = np.argmin(np.sum(
        (sources[wall_id_i, :, :]-difference_source)**2, axis=-1))
    receiver_idx = np.argmin(np.sum(
        (receivers[wall_id_i, :]-difference_receiver)**2, axis=-1))
    return scattering[scattering_index[wall_id_i],
        source_idx, receiver_idx, :]


def get_scattering_data_source(
        pos_h:np.ndarray, pos_i:np.ndarray,
        sources:np.ndarray, wall_id_i:np.ndarray,
        scattering:np.ndarray, scattering_index:np.ndarray):
    """Get scattering data depending on previous, current position.

    Parameters
    ----------
    pos_h : np.ndarray
        previous position of shape (3)
    pos_i : np.ndarray
        current position of shape (3)
    sources : np.ndarray
        source directions of all walls of shape (n_walls, n_sources, 3)
    wall_id_i : np.ndarray
        current wall id to get write directional data
    scattering : np.ndarray
        scattering data of shape (n_scattering, n_sources, n_receivers, n_bins)
    scattering_index : np.ndarray
        index of the scattering data of shape (n_walls)

    Returns
    -------
    scattering_factor: float
        scattering factor from directivity

    """
    difference_source = pos_h-pos_i
    difference_source /= np.linalg.norm(difference_source)
    source_idx = np.argmin(np.sum(
        (sources[wall_id_i, :, :]-difference_source)**2, axis=-1))
    return scattering[scattering_index[wall_id_i], source_idx]

if numba is not None:
    _add_directional = numba.njit(parallel=True)(_add_directional)
    _energy_exchange_init_energy = numba.njit()(_energy_exchange_init_energy)
    _collect_receiver_energy = numba.njit()(_collect_receiver_energy)
    _energy_exchange = numba.njit()(_energy_exchange)
    _form_factors_with_directivity_dim = numba.njit(parallel=True)(
        _form_factors_with_directivity_dim)
    _form_factors_with_directivity = numba.njit(parallel=True)(
        _form_factors_with_directivity)
    get_scattering_data_receiver_index = numba.njit()(
        get_scattering_data_receiver_index)
    get_scattering_data = numba.njit()(get_scattering_data)
    get_scattering_data_source = numba.njit()(get_scattering_data_source)



