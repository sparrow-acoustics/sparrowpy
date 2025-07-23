"""Module for the radiosity simulation."""
import numpy as np
import deepdiff
import pyfar as pf
import sparrowpy.form_factor.universal as form_factor
from sparrowpy import ( geometry, sound_object )
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
    _n_patches: int
    _patch_to_wall_ids: np.ndarray

    # geometrical data
    _visibility_matrix: np.ndarray
    _visible_patches: np.ndarray
    _form_factors: np.ndarray
    _form_factors_tilde: np.ndarray

    # general data for material and medium data
    _frequencies: np.ndarray
    _brdf: np.ndarray
    _brdf_index: np.ndarray
    _brdf_incoming_directions: list[pf.Coordinates]
    _brdf_outgoing_directions: list[pf.Coordinates]
    _patch_2_brdf_outgoing_index: np.ndarray

    _air_attenuation: np.ndarray
    _speed_of_sound: float

    # etc metadata
    _etc_time_resolution: float
    _etc_duration: float

    # etc results
    _distance_patches_to_source: np.ndarray
    _energy_init_source: np.ndarray
    _energy_exchange_etc: np.ndarray


    def __init__(
            self,
            walls_points:np.ndarray,
            walls_normal:np.ndarray,
            walls_up_vector:np.ndarray,
            patches_points:np.ndarray,
            n_patches:int,
            patch_to_wall_ids:np.ndarray,
            visibility_matrix:np.ndarray=None,
            visible_patches:np.ndarray=None,
            form_factors:np.ndarray=None,
            form_factors_tilde:np.ndarray=None,
            frequencies:np.ndarray=None,
            brdf:list[np.ndarray]=None,
            brdf_index:np.ndarray=None,
            brdf_incoming_directions:list[pf.Coordinates]=None,
            brdf_outgoing_directions:list[pf.Coordinates]=None,
            patch_2_brdf_outgoing_index:np.ndarray=None,
            air_attenuation:np.ndarray=None,
            speed_of_sound:float=None,
            etc_time_resolution:float=None,
            etc_duration:float=None,
            distance_patches_to_source:np.ndarray=None,
            energy_init_source:np.ndarray=None,
            energy_exchange_etc:np.ndarray=None):
        """Create a Radiosity object for directional implementation.

        Parameters
        ----------
        walls_points : np.ndarray
            edge points of all walls in cartesian of shape
            (n_walls, n_points, 3)
        walls_normal : np.ndarray
            normals of all walls of shape (n_walls, 3)
        walls_up_vector : np.ndarray
            uo vector of all walls of shape (n_walls, 3)
        patches_points : np.ndarray
            edge points of all patches in cartesian of shape
            (n_patches, n_points, 3)
        n_patches : int
            number of patches
        patch_to_wall_ids : np.ndarray
            maps each patch to a wall of shape (n_patches)
        visibility_matrix : np.ndarray, optional
            patch to patch boolean visibility matrix, by default None.
        visible_patches : np.ndarray, optional
            list of all patches which are visible, by default None
        form_factors : np.ndarray, optional
            geometrical form factor from each patch to each other patch,
            by default None.
        form_factors_tilde : np.ndarray, optional
            Form factor including air attenuation and BRDF of shape
            (n_patches, n_patches, n_outgoing_directions, n_bins), by
            default None
        frequencies : np.ndarray, optional
            frequency vector used for the simulation. , by default None
        brdf : list[np.ndarray], optional
            brdf in its raw postprocessed data format, by default None
        brdf_index : np.ndarray, optional
            maps brdfs to walls, must be of shape (n_walls, ), by default None
        brdf_incoming_directions : list[pf.Coordinates], optional
            incoming direction of brdfs per wall, by default None
        brdf_outgoing_directions : list[pf.Coordinates], optional
            outgoing directions of brdfs per wall, by default None
        patch_2_brdf_outgoing_index: np.ndarray
            map of patch positions to relative scattering directions indices
        air_attenuation : np.ndarray, optional
            air attenuation coefficients for each frequency, needs to be of
            shape (n_bins), by default None
        speed_of_sound : float, optional
            speed of sound in m/s, by default None
        etc_time_resolution : float, optional
            time resolution fo the etc, by default None
        etc_duration : float, optional
            duration fo the etc in seconds, by default None
        distance_patches_to_source : np.ndarray, optional
            distance from the source to each patch, need to be of shape
            (n_patches), by default None
        energy_init_source : np.ndarray, optional
            initial energy from source to patches,
            need to be of shape (n_patches, n_outgoing_directions, n_bins),
            by default None
        energy_exchange_etc : np.ndarray, optional
            etc of the energy exchange, must be of shape
            (n_patches, n_outgoing_directions, n_bins, n_samples),
            by default None
        """
        # convert inputs
        walls_points = np.atleast_3d(walls_points)
        walls_up_vector = np.atleast_2d(walls_up_vector)
        walls_normal = np.atleast_2d(walls_normal)
        patches_points = np.atleast_3d(patches_points)
        patch_to_wall_ids = np.atleast_1d(np.array(
            patch_to_wall_ids, dtype=int))
        if frequencies is not None:
            frequencies = np.array(frequencies)
        if visible_patches is not None:
            visible_patches = np.array(visible_patches)
        if visibility_matrix is not None:
            visibility_matrix = np.array(visibility_matrix)
        if form_factors is not None:
            form_factors = np.array(form_factors)
        if form_factors_tilde is not None:
            form_factors_tilde = np.array(form_factors_tilde)
        if patch_2_brdf_outgoing_index is not None:
            patch_2_brdf_outgoing_index = np.array(patch_2_brdf_outgoing_index,
                                                   dtype=np.int64)
        if brdf is not None:
            brdf = [np.array(b) for b in brdf]
        if air_attenuation is not None:
            air_attenuation = np.array(air_attenuation)
        if speed_of_sound is not None:
            speed_of_sound = float(speed_of_sound)
        if etc_time_resolution is not None:
            etc_time_resolution = float(etc_time_resolution)
        if etc_duration is not None:
            etc_duration = float(etc_duration)
        if distance_patches_to_source is not None:
            distance_patches_to_source = np.array(distance_patches_to_source)
        if energy_init_source is not None:
            energy_init_source = np.array(energy_init_source)
        if energy_exchange_etc is not None:
            energy_exchange_etc = np.array(energy_exchange_etc)

        self._walls_points = walls_points
        self._walls_normal = walls_normal
        self._walls_up_vector = walls_up_vector
        self._patches_points = patches_points
        self._n_patches = n_patches
        self._patch_to_wall_ids = patch_to_wall_ids

        # geometrical data
        self._visibility_matrix = visibility_matrix
        self._visible_patches = visible_patches
        self._form_factors = form_factors
        self._form_factors_tilde = form_factors_tilde

        # general data for material and medium data
        self._frequencies = frequencies
        self._brdf = brdf
        self._brdf_index = brdf_index
        self._brdf_incoming_directions = brdf_incoming_directions
        self._brdf_outgoing_directions = brdf_outgoing_directions
        self._patch_2_brdf_outgoing_index = patch_2_brdf_outgoing_index

        self._air_attenuation = air_attenuation
        self._speed_of_sound = speed_of_sound

        # etc metadata
        self._etc_time_resolution = etc_time_resolution
        self._etc_duration = etc_duration

        # etc results
        self._distance_patches_to_source = distance_patches_to_source
        self._energy_init_source = energy_init_source
        self._energy_exchange_etc = energy_exchange_etc

        self.check()

    def check(self):
        """Check the input data for consistency."""

        n_walls = self._walls_points.shape[0]
        if (self._walls_points.shape[0] != n_walls) or \
                (self._walls_points.shape[2] != 3):
            raise ValueError(
                "Walls need to be of shape (n_walls, n_points, 3)")
        if self._walls_up_vector.shape != (n_walls, 3):
            raise ValueError(
                "Up vector of walls need to be of shape (n_walls, 3)")
        if self._walls_normal.shape != (n_walls, 3):
            raise ValueError(
                "Normal of walls need to be of shape (n_walls, 3)")

        # input checks Patches
        if (self._patches_points.shape[0] != self.n_patches) or \
                (self._patches_points.shape[2] != 3):
            raise ValueError(
                "Patches need to be of shape (n_patches, n_points, 3)")
        if self._patch_to_wall_ids.shape != (self.n_patches,):
            raise ValueError(
                "patch_to_wall_ids need to be of shape (n_patches,)")
        if any(i not in np.arange(
                n_walls) for i in set(self._patch_to_wall_ids)):
            raise ValueError(
                "patch_to_wall_ids does contain other ids than range(n_walls)")
        if any(i not in set(
                self._patch_to_wall_ids) for i in np.arange(n_walls)):
            raise ValueError(
                "patch_to_wall_ids does contain other ids than range(n_walls)")

        # check frequencies
        n_bins = 1
        if self._frequencies is not None:
            if len(self._frequencies.shape) != 1:
                raise ValueError(
                    "Frequencies need to be of shape (n_bins,)")
            n_bins = self._frequencies.size

        # check form factors
        if self._form_factors is not None:
            if self._form_factors.shape != (self.n_patches, self.n_patches):
                raise ValueError(
                    "form_factors need to be of shape (n_patches, n_patches)")

        # check brdf
        n_outgoing_directions = 1
        if self._brdf_index is not None:
            if len(self._brdf_index) != n_walls:
                raise ValueError(
                    "brdf_index need to be of shape (n_walls,)")
        if self._brdf_incoming_directions is not None:
            if any(not isinstance(i, pf.Coordinates) \
                   for i in self._brdf_incoming_directions):
                raise ValueError(
                    "brdf_incoming_directions need to be a list of type "
                    "pf.Coordinates")
        if self._brdf_outgoing_directions is not None:
            if any(not isinstance(i, pf.Coordinates) \
                   for i in self._brdf_outgoing_directions):
                raise ValueError(
                    "brdf_outgoing_directions need to be a list of type "
                    "pf.Coordinates")
            n_outgoing_directions = self._brdf_outgoing_directions[0].csize

        # check form_factors_tilde
        if self._form_factors_tilde is not None:
            if self._form_factors_tilde.shape != (
                    self.n_patches, self.n_patches,
                    n_outgoing_directions, n_bins):
                raise ValueError(
                    "form_factors_tilde need to be of shape "
                    "(n_patches, n_patches, n_outgoing_directions, n_bins)")

        # check air_attenuation
        if self._air_attenuation is not None:
            if len(self._air_attenuation.shape) != 1:
                raise ValueError(
                    "Air attenuation need to be of shape (n_bins,)")
            if self._air_attenuation.shape[0] != n_bins:
                raise ValueError(
                    "Air attenuation need to be of shape (n_bins,)")

        # check speed_of_sound
        if self._speed_of_sound is not None:
            if self._speed_of_sound <= 0:
                raise ValueError(
                    "Speed of sound must be positive and non-zero")

        # check etc_time_resolution
        if self._etc_time_resolution is not None:
            if self._etc_time_resolution <= 0:
                raise ValueError(
                    "Time resolution must be positive and non-zero")

        # check etc_duration
        if self._etc_duration is not None:
            if self._etc_duration <= 0:
                raise ValueError(
                    "Duration must be positive and non-zero")

        # check distance_patches_to_source
        if self._distance_patches_to_source is not None:
            if self._distance_patches_to_source.shape != (self.n_patches,):
                raise ValueError(
                    "distance_patches_to_source need to be of shape "
                    "(n_patches,)")

        # check energy_init_source
        if self._energy_init_source is not None:
            if self._energy_init_source.shape != (
                    self.n_patches, n_outgoing_directions, n_bins):
                raise ValueError(
                    "energy_init_source need to be of shape "
                    "(n_patches, n_outgoing_directions, n_bins)")

        # check energy_exchange_etc
        if self._energy_exchange_etc is not None:
            n_samples = int(
                self._etc_duration/self._etc_time_resolution)
            if self._energy_exchange_etc.shape != (
                    self.n_patches,
                    n_outgoing_directions, n_bins, n_samples):
                raise ValueError(
                    "energy_exchange_etc need to be of shape "
                    "(n_patches, n_outgoing_directions, n_bins, n_samples)")


    @classmethod
    def from_polygon(
            cls, polygon_list, patch_size):
        """Create a Radiosity object for directional scattering coefficients.

        Parameters
        ----------
        polygon_list : list[Polygon]
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
            patches_points, _,
            n_patches, patch_to_wall_ids) = geometry._process_patches(
            walls_points, walls_normal, patch_size, len(polygon_list))
        # create radiosity object
        return cls(
            walls_points, walls_normal, walls_up_vector,
            patches_points, n_patches,
            patch_to_wall_ids)

    def bake_geometry(self):
        """Bake the geometry by calculating all the form factors.

        """
        # Check the visibility between patches.
        self._visibility_matrix = geometry._check_patch2patch_visibility(
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

            # preload patch_2_brdf_outgoing_index map with invalid entries
            self._patch_2_brdf_outgoing_index = (
                        receivers_array.shape[1] *
                                np.ones((self.n_patches,self.n_patches),dtype=np.int64))

            for j in range(self.n_patches):
                vis = np.where(
                    (self.visibility_matrix+self.visibility_matrix.T)[:,j])
                self._patch_2_brdf_outgoing_index[vis,j]=get_scattering_data_receiver_index(
                        pos_i=self.patches_center[vis],pos_j=self.patches_center[j],
                        receivers=receivers_array,
                        wall_id_i=self._patch_to_wall_ids[vis],
                    )
        else:
            sources_array = None
            receivers_array = None
            scattering_index = None
            scattering = None
            self._patch_2_brdf_outgoing_index = np.zeros(
                                        (self.n_patches,self.n_patches),
                                        dtype=np.int64)

        n_bins = 1 if self._frequencies is None else self.n_bins

        self._form_factors_tilde = \
            _form_factors_with_directivity_dim(
            self.visibility_matrix, self.form_factors, n_bins,
            self.patches_center, self.patches_area,
            self._air_attenuation,
            self._patch_to_wall_ids, scattering,
            scattering_index,
            sources_array, receivers_array)


    def init_source_energy(
            self, source):
        """Initialize the source energy.

        Parameters
        ----------
        source : pf.Coordinates, sparrowpy.sound_object.SoundSource
            definition of the source position for Coordinates object and
            orientation and directivity for SoundSource object. If no
            directivity is given, the directivity is set to 1 for all
            frequencies.

        """
        if isinstance(source, pf.Coordinates):
            if source.cshape != (1, ):
                raise ValueError('just one source position is allowed.')
            source_position = source.cartesian[0]
        elif isinstance(source, sound_object.SoundSource):
            source_position = source.position
        self._source = source


        patch_to_wall_ids = self._patch_to_wall_ids
        if self._brdf_incoming_directions is None:
            frequencies = np.array([0]) if self._frequencies is None else \
                self._frequencies
            self.set_wall_brdf(
                np.arange(self.n_walls),
                pf.FrequencyData(np.ones_like(frequencies), frequencies),
                pf.Coordinates(0, 0, 1, weights=1),
                pf.Coordinates(0, 0, 1, weights=1))
            self._frequencies = frequencies
        if self._air_attenuation is None:
            frequencies = np.array([0]) if self._frequencies is None else \
                self._frequencies
            self.set_air_attenuation(
                pf.FrequencyData(np.zeros_like(frequencies), frequencies))
            self._frequencies = frequencies
        n_bins = self.n_bins
        vi = np.array(
            [s.cartesian for s in self._brdf_incoming_directions])
        vo = np.array(
            [s.cartesian for s in self._brdf_outgoing_directions])
        brdf = np.array(self._brdf)
        brdf_index = self._brdf_index
        patches_center = self.patches_center
        source_visibility = geometry._check_point2patch_visibility(
                                        eval_point=source_position,
                                        patches_center=patches_center,
                                        surf_points=self.walls_points,
                                        surf_normal=self.walls_normal)
        self._source_visibility = source_visibility
        energy_0, distance_0 = form_factor._source2patch_energy_universal(
            source_position, patches_center, self.patches_points,
            source_visibility,
            self._air_attenuation, n_bins)

        # of shape (n_patches, n_directions, n_bins)
        energy_0_dir = _add_directional(
            energy_0, source_position,
            patches_center, n_bins, patch_to_wall_ids,
            vi, vo, brdf, brdf_index)

        # add directivity if given
        if isinstance(source, sound_object.SoundSource):
            n_patches = patches_center.shape[0]
            n_directions = vo.shape[1]

            if source.directivity is not None:
                directivity = np.zeros((n_patches, n_directions, n_bins))
                for i_frequency in range(n_bins):
                    directivity_local = np.real(source.get_directivity(
                        patches_center, self._frequencies[i_frequency]))
                    if n_directions == 1:
                        directivity[:, :, i_frequency] = directivity_local[
                            :, np.newaxis]
                    else:
                        directivity[:, :, i_frequency] = np.repeat(
                            directivity_local[..., np.newaxis],
                            n_directions,
                            axis=-1)
                energy_0_dir *= directivity
            else:
                energy_0_dir *= 1

        self._energy_init_source = energy_0_dir
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

        energy_0_dir = self._energy_init_source

        if self._energy_exchange_etc is None or recalculate:
            # energy exchange etc
            # of shape (n_patches, n_directions, n_bins, n_samples)
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
                    self._patch_2_brdf_outgoing_index,
                    speed_of_sound, etc_time_resolution,
                    max_reflection_order,
                    self._visible_patches)

        self._etc_time_resolution = float(etc_time_resolution)
        self._speed_of_sound = float(speed_of_sound)
        self._etc_duration = float(etc_duration)

    def collect_energy_receiver_mono(self, receivers, direct_sound=False):
        """Collect the energy at the receivers.

        Parameters
        ----------
        receivers : pf.Coordinates
            receiver Coordinates in of cshape (n_receivers).
        direct_sound : bool, optional
            If True, the direct sound is collected as well, by default False.
            The direct sound includes spreading loss, air attenuation and
            source directivity.

        Returns
        -------
        etc : pf.TimeData
            energy collected at the receiver in cshape
            (n_receivers, n_bins)
        """
        if not isinstance(direct_sound, bool):
            raise ValueError(
                "direct_sound must be of type boolean")
        etc = self.collect_energy_receiver_patchwise(receivers)
        etc.time = np.sum(etc.time, axis=1)

        if direct_sound:
            direct_sound, n_sample_delay = self.calculate_direct_sound(
                receivers)

            # add the direct sound to the etc
            i_receivers = np.arange(len(n_sample_delay))
            etc.time[i_receivers, :, n_sample_delay] += direct_sound

        return etc


    def calculate_direct_sound(self, receivers):
        """Calculate the direct sound at the receivers.

        It includes the spreading loss, air attenuation and
        source directivity.

        Parameters
        ----------
        receivers : pf.Coordinates
            receiver Coordinates in of cshape (n_receivers).

        Returns
        -------
        direct_sound : np.ndarray
            energy of the direct sound at the receivers in shape
            (n_receivers, n_bins)
        n_sample_delay : np.ndarray
            number of samples for the delay at the receivers in shape
            (n_receivers, )
        """
        if not isinstance(receivers, pf.Coordinates):
            raise ValueError(
                "Receiver positions must be of type pf.Coordinates")

        # calculate distance from source to receivers
        if isinstance(self._source, pf.Coordinates):
            source_position = self._source
        else:
            source_position = pf.Coordinates(*self._source.position)
        r = (receivers-source_position).radius

        # calculate spreading loss for direct sound
        direct_sound = np.ones(
            (receivers.cshape[0], self.n_bins), dtype=float)
        direct_sound *= (1/(4 * np.pi * r**2))[:, np.newaxis]

        # add air attenuation
        if self._air_attenuation is not None:
            for i in range(self.n_bins):
                direct_sound[:, i] *= np.exp(
                    -self._air_attenuation[i] * r)

        # add source directivity
        if isinstance(self._source, sound_object.SoundSource):
            for i in range(self.n_bins):
                direct_sound[:, i] *= np.real(self._source.get_directivity(
                    np.squeeze(receivers.cartesian), self._frequencies[i]))

        # calculate the number of samples for the delay
        n_sample_delay = np.array(
            r/self.speed_of_sound/self._etc_time_resolution, dtype=int)

        return direct_sound, n_sample_delay


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

        receiver_visibility=np.empty((n_receivers,n_patches),dtype=bool)

        for i in range(n_receivers):
            patches_receiver_distance = patches_center - receiver_pos[i]

            receiver_visibility[i] = geometry._check_point2patch_visibility(
                                        eval_point=receiver_pos[i],
                                        patches_center=patches_center,
                                        surf_points=self.walls_points,
                                        surf_normal=self.walls_normal)

            # geometrical weighting
            patch_receiver_energy=form_factor._patch2receiver_energy_universal(
                    receiver_pos[i], patches_points, receiver_visibility[i])

            # access histograms with correct scattering weighting
            receivers_array = np.array(
                [s.cartesian for s in self._brdf_outgoing_directions])

            receiver_idx = get_scattering_data_receiver_index(
                patches_center, receiver_pos[i], receivers_array,
                self._patch_to_wall_ids)

            assert receiver_idx.shape[0] == self.n_patches
            assert len(receiver_idx.shape) == 1

            for k in range(n_patches):
                E_matrix[k,:]= (
                    self._energy_exchange_etc[k,int(receiver_idx[k]),:]
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
        air_attenuation : pyfar.FrequencyData
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
        if self._frequencies is None:
            self._frequencies = frequencies
        else:
            assert self._frequencies.size == frequencies.size, \
                "Number of frequency bins do not match"
            assert (self._frequencies == frequencies).all(), \
                "Frequencies do not match"

    def write(self, filename, compress=True):
        """Write the object to a far file."""
        pf.io.write(filename, compress=compress, **self.to_dict())

    @classmethod
    def from_read(cls, filename):
        """Read the object to a far file."""
        data = pf.io.read(filename)
        for key, value in data.items():
            if isinstance(value, str) and value == 'None':
                data[key] = None
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Convert the object to a dictionary."""
        dict_out = {
            'walls_points': self._walls_points,
            'walls_normal': self._walls_normal,
            'walls_up_vector': self._walls_up_vector,
            'patches_points': self._patches_points,
            'n_patches': self._n_patches,
            'patch_to_wall_ids': self._patch_to_wall_ids,
            'visibility_matrix': self._visibility_matrix,
            'visible_patches': self._visible_patches,
            'form_factors': self._form_factors,
            'form_factors_tilde': self._form_factors_tilde,
            'frequencies': self._frequencies,
            'brdf': self._brdf,
            'brdf_index': self._brdf_index,
            'brdf_incoming_directions': self._brdf_incoming_directions,
            'brdf_outgoing_directions': self._brdf_outgoing_directions,
            'patch_2_brdf_outgoing_index': self._patch_2_brdf_outgoing_index,
            'air_attenuation': self._air_attenuation,
            'speed_of_sound': self._speed_of_sound,
            'etc_time_resolution': self._etc_time_resolution,
            'etc_duration': self._etc_duration,
            'distance_patches_to_source': self._distance_patches_to_source,
            'energy_init_source': self._energy_init_source,
            'energy_exchange_etc': self._energy_exchange_etc,
        }
        for key, value in dict_out.items():
            if value is None:
                dict_out[key] = 'None'
            elif isinstance(value, np.ndarray):
                dict_out[key] = value.tolist()
        return dict_out

    def __eq__(self, other):
        """Check for equality of two objects."""
        if not isinstance(other, DirectionalRadiosityFast):
            return False
        return not deepdiff.DeepDiff(self.to_dict(), other.to_dict())


    @classmethod
    def from_dict(cls, input_dict: dict):
        """Create an object from a dictionary. Used for read write."""
        obj = cls(**input_dict)
        return obj

    @property
    def n_bins(self):
        """Return the number of frequency bins."""
        if self._frequencies is None:
            return None
        return self._frequencies.shape[0]

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
        return self._walls_normal[self._patch_to_wall_ids]

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
            * np.real(scattering_factor)

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
        form_factors_tilde, patch_2_out_directions,
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
    patch_2_out_directions: np.ndarray
        patchwise map of patch centers to
        scattering outgoing directions of shape (n_patches,n_patches)
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

                dir_id = patch_2_out_directions[i,j]

                n_delay_samples = int(
                    distance_ij[i, j]/speed_of_sound/histogram_time_resolution)
                if n_delay_samples > 0:
                    E_matrix[current_index, j, :, :, n_delay_samples:] += \
                        form_factors_tilde[i, j] * E_matrix[
                            current_index-1, i, dir_id, :, :-n_delay_samples]
                else:
                    E_matrix[current_index, j, :, :, :] += form_factors_tilde[
                        i, j] * E_matrix[current_index-1, i, dir_id, :, :]
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
    """Get scattering receiver index based on current and next position.

    Parameters
    ----------
    pos_i : np.ndarray
        current position of shape (n,3)
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
        difference_receiver = pos_j-pos_i[i]

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



