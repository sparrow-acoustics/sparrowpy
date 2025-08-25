"""Module for the radiosity simulation."""
import matplotlib as mpl
import numpy as np
import pyfar as pf
import sofar as sf
from sparrowpy.geometry import Polygon, SoundSource


class PatchesKang(Polygon):
    """Class representing patches of a polygon."""

    patches: list[Polygon]
    other_wall_ids: list[int]  # ids of the patches
    form_factors: np.ndarray
    E_matrix: np.ndarray
    wall_id: int
    scattering: np.ndarray
    absorption: np.ndarray
    n_bins: int
    max_size: float
    E_sampling_rate: int
    E_n_samples: int
    sound_attenuation_factor: np.ndarray

    def __init__(
            self, polygon, max_size, other_wall_ids, wall_id,
            scattering=1, absorption=0.1, sound_attenuation_factor=0,
            E_matrix=None):
        """Init Directional Patches.

        Parameters
        ----------
        polygon : Polygon
            Wall that should be distributed into patches.
        max_size : float
            maximal patchsize in meters.
        other_wall_ids : list[int]
            Ids of the other walls.
        wall_id : int
            id of this Patch
        scattering : np.ndarray, optional
            Scattering coefficient of this wall for each freqeuncy
            bin, by default 1
        absorption : np.ndarray, optional
            Absorption coefficient of this wall for each freqeuncy
            bin, by default 0.1
        sound_attenuation_factor : _type_, optional
            Air attenuation factor for each frequncy bin, by default None
        E_matrix : np.ndarray, optional
            Energy exchange results, if already calcualted , by default None

        """
        Polygon.__init__(self, polygon.pts, polygon.up_vector, polygon.normal)
        min_point = np.min(polygon.pts, axis=0)
        max_point = np.max(polygon.pts, axis=0)
        size = max_point - min_point
        patch_nums = np.array([int(n) for n in size/max_size])
        real_size = size/patch_nums
        self.scattering = np.atleast_1d(scattering)
        self.absorption = np.atleast_1d(absorption)
        self.sound_attenuation_factor = np.atleast_1d(sound_attenuation_factor)
        assert self.scattering.size == self.absorption.size
        assert self.scattering.size == self.sound_attenuation_factor.size
        self.n_bins = self.absorption.size

        if patch_nums[2] == 0:
            x_idx = 0
            y_idx = 1
        if patch_nums[1] == 0:
            x_idx = 0
            y_idx = 2
        if patch_nums[0] == 0:
            x_idx = 1
            y_idx = 2

        x_min = np.min(polygon.pts.T[x_idx])
        y_min = np.min(polygon.pts.T[y_idx])

        patches = []
        for i_x in range(patch_nums[x_idx]):
            for i_y in range(patch_nums[y_idx]):
                points = polygon.pts.copy()
                points[0, x_idx] = x_min + i_x * real_size[x_idx]
                points[0, y_idx] = y_min + i_y * real_size[y_idx]
                points[1, x_idx] = x_min + (i_x+1) * real_size[x_idx]
                points[1, y_idx] = y_min + i_y * real_size[y_idx]
                points[3, x_idx] = x_min + i_x * real_size[x_idx]
                points[3, y_idx] = y_min + (i_y+1) * real_size[y_idx]
                points[2, x_idx] = x_min + (i_x+1) * real_size[x_idx]
                points[2, y_idx] = y_min + (i_y+1) * real_size[y_idx]

                patch = Polygon(points, polygon.up_vector, polygon.normal)
                patches.append(patch)

        self.patches = patches
        self.other_wall_ids = np.atleast_1d(np.array(
            other_wall_ids, dtype=int))

        self.wall_id = wall_id

        self.max_size = max_size

        if E_matrix is not None:
            self.E_matrix = np.array(E_matrix)
            self.E_n_samples = self.E_matrix.shape[3]

    def to_dict(self) -> dict:
        """Convert this object to dictionary. Used for read write."""
        return {
            'pts': self.pts,
            'up_vector': self.up_vector,
            'normal': self.normal,
            'max_size': self.max_size,
            'other_wall_ids': self.other_wall_ids,
            'wall_id': self.wall_id,
            'scattering': self.scattering,
            'absorption': self.absorption,
            'sound_attenuation_factor': self.sound_attenuation_factor,
            'E_matrix': self.E_matrix,
        }

    @classmethod
    def from_dict(cls, input_dict: dict):
        """Create an object from a dictionary. Used for read write."""
        return cls(
            Polygon.from_dict(input_dict),
            input_dict['max_size'],
            input_dict['other_wall_ids'],
            input_dict['wall_id'],
            input_dict['scattering'],
            input_dict['absorption'],
            input_dict['sound_attenuation_factor'],
            E_matrix=input_dict['E_matrix'])

    def plot(self, ax: mpl.axes.Axes = None, color=None):
        """Plot the patches."""
        for patch in self.patches:
            patch.plot_point(ax, color)
        self.plot_view_up(ax, color)

    def init_energy_exchange(
            self, max_order_k, ir_length_s, source, sampling_rate,
            speed_of_sound):
        """Initialize the energy exchange Matrix with source energy.

        It init the matrix self.E_matrix and add source energy after (6).

        Parameters
        ----------
        max_order_k : int
            max order of energy exchange iterations.
        ir_length_s : float
            length of the impulse response in seconds.
        source : SoundSource
            sound source with ``sound_power`` and ``position``
        sampling_rate : int, optional
            Sampling rate of impulse response.
        speed_of_sound : float, optional
            speed of sound in m/s.

        """
        self.E_sampling_rate = sampling_rate
        self.E_n_samples = int(ir_length_s*sampling_rate)

        self.E_matrix = np.zeros((
            self.n_bins,
            max_order_k+1,  # order of energy exchange
            len(self.patches),  # receiver ids of patches
            self.E_n_samples,  # impulse response G_k(t)_receiverpatch
            ))

        for i_receiver, receiver_patch in enumerate(self.patches):
            source_pos = source.position.copy()
            receiver_pos = receiver_patch.center.copy()

            distance = np.linalg.norm(receiver_pos-source_pos)
            delay_seconds = distance/speed_of_sound
            delay_samples = int(delay_seconds*self.E_sampling_rate)

            if np.abs(receiver_patch.normal[2]) > 0.99:
                i = 2
                indexes = [0, 1, 2]
            elif np.abs(receiver_patch.normal[1]) > 0.99:
                indexes = [2, 0, 1]
                i = 1
            elif np.abs(receiver_patch.normal[0]) > 0.99:
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
            dd_l = receiver_patch.size[indexes[0]]
            dd_m = receiver_patch.size[indexes[1]]
            S_x = source_pos[indexes[0]]
            S_y = source_pos[indexes[1]]
            S_z = source_pos[indexes[2]]
            energy = _init_energy_exchange(
                dl, dm, dn, dd_l, dd_m, S_x, S_y, S_z,
                source.sound_power, self.absorption, distance,
                self.sound_attenuation_factor, self.n_bins)
            self.E_matrix[:, 0, i_receiver, delay_samples] += energy


    def calculate_energy_exchange(
            self, patches_list, current_order_k, speed_of_sound,
            E_sampling_rate):
        """Calculate the energy exchange for a given order.

        It implements formula 18 and save the result in self.E_matrix.

        Parameters
        ----------
        patches_list : list[Patches]
            list of all patches
        current_order_k : int
            Order k
        speed_of_sound : int, optional
            speed of sound in m/s.
        E_sampling_rate : int, optional
            Sampling rate of histogram.

        """
        k = current_order_k # real k 1 .. max_k
        for i_receiver, receiver_patch in enumerate(self.patches):
            receiver = receiver_patch.center

            for wall_id in self.other_wall_ids:
                wall = patches_list[wall_id]
                for i_source, source_patch in enumerate(wall.patches):
                    source = source_patch.center

                    # distance between source and receiver patches
                    distance = np.linalg.norm(receiver-source)
                    delay_seconds = distance/speed_of_sound

                    # sample delay for the given distance
                    delay_samples = int(delay_seconds*E_sampling_rate)

                    for i_frequency in range(self.n_bins):
                        # access energy matrix of source patch, previous order
                        A_k_minus_1 = wall.E_matrix[
                            i_frequency, k-1, i_source, :]

                        # delay IR by delay_samples
                        A_k_minus1_delay = _add_delay(
                            A_k_minus_1, delay_samples)

                        # find form factor of source and receiver patches
                        form_factor = wall.get_form_factor(
                            patches_list, i_source, self.wall_id, i_receiver)
                        alpha = self.absorption[i_frequency]

                        # multiply delayed IR by form factor
                        energy = A_k_minus1_delay * form_factor \
                            * self.scattering[i_frequency] * \
                                (1-alpha) * np.exp(
                                    -self.sound_attenuation_factor[
                                        i_frequency] * distance)

                        # add energy to energy matrix of self
                        self.E_matrix[i_frequency, k, i_receiver, :] += energy
                        #

    def calculate_form_factor(self, patches_list) -> None:
        """Calculate the form factors between patches.

        Parameters
        ----------
        patches_list : list of patches
            List of patches.
        M : float, optional
            Air attenuation factor in Np/m, by default 0
        alpha : float, optional
            absorption coefficient of wall, by default 0.1

        """
        num_other_patches = np.sum([
            len(patches_list[i].patches) for i in self.other_wall_ids])
        self.form_factors = np.zeros((len(self.patches), num_other_patches))
        for i_source, source_patch in enumerate(self.patches):
            i_receiver_offset = 0
            for receiver_wall_id in self.other_wall_ids:
                receiver_wall = patches_list[receiver_wall_id]
                for i_receiver, receiver_patch in enumerate(
                        receiver_wall.patches):

                    difference = np.abs(
                        receiver_wall.center-self.center)
                    # calculation of form factors
                    dot_product = np.dot(
                        receiver_patch.normal, source_patch.normal)

                    dd_l = self.max_size
                    dd_m = self.max_size
                    dd_n = self.max_size

                    if dot_product == 0:  # orthogonal
                        source_center = source_patch.center
                        receiver_center = receiver_patch.center

                        if np.abs(source_patch.normal[0]) > 1e-5:
                            idx_source = {2, 1}
                            dl = source_center[2]
                            dm = source_center[1]
                        elif np.abs(source_patch.normal[1]) > 1e-5:
                            idx_source = {2, 0}
                            dl = source_center[2]
                            dm = source_center[0]
                        elif np.abs(source_patch.normal[2]) > 1e-5:
                            idx_source = {0, 1}
                            dl = source_center[1]
                            dm = source_center[0]

                        if np.abs(receiver_patch.normal[0]) > 1e-5:
                            idx_receiver = {2, 1}
                            dl_prime = receiver_center[1]
                            dn_prime = receiver_center[2]
                        elif np.abs(receiver_patch.normal[1]) > 1e-5:
                            idx_receiver = {0, 2}
                            dl_prime = receiver_center[0]
                            dn_prime = receiver_center[2]
                        elif np.abs(receiver_patch.normal[2]) > 1e-5:
                            idx_receiver = {0, 1}
                            dl_prime = receiver_center[1]
                            dn_prime = receiver_center[0]
                        idx_l = tuple(idx_receiver.intersection(idx_source))
                        idx_s = tuple(idx_source.difference(set(idx_l)))
                        idx_r = tuple(idx_receiver.difference(set(idx_l)))
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
                            dl = receiver_patch.center[1]
                            dm = receiver_patch.center[0]
                            dn = receiver_patch.center[2]
                            dl_prime = source_patch.center[1]
                            dm_prime = source_patch.center[0]
                            dn_prime = source_patch.center[2]
                        elif difference[1] > 1e-5:
                            dl = receiver_patch.center[0]
                            dm = receiver_patch.center[1]
                            dn = receiver_patch.center[2]
                            dl_prime = source_patch.center[0]
                            dm_prime = source_patch.center[1]
                            dn_prime = source_patch.center[2]
                        elif difference[2] > 1e-5:
                            dl = receiver_patch.center[1]
                            dm = receiver_patch.center[2]
                            dn = receiver_patch.center[0]
                            dl_prime = source_patch.center[1]
                            dm_prime = source_patch.center[2]
                            dn_prime = source_patch.center[0]
                        else:
                            raise AssertionError()

                        d = np.sqrt(
                            ( (dl - dl_prime) ** 2 ) +
                            ( (dn - dn_prime) ** 2 ) +
                            ( (dm - dm_prime) ** 2 ) )
                        # Equation 16
                        ff =  ( dd_l * dd_n * ( (
                            dm-dm_prime) ** 2 ) ) / ( np.pi * ( d**4 ) )

                    index_rec = i_receiver + i_receiver_offset
                    self.form_factors[i_source, index_rec] = ff

                i_receiver_offset += len(receiver_wall.patches)

    def get_form_factor(
            self, patches_list, source_path_id, receiver_wall_id,
            receiver_patch_id):
        """Return form factor.

        Parameters
        ----------
        patches_list : list of Patches
            patches list
        source_path_id : int
            _description_
        receiver_wall_id : int
            receiver wall id
        receiver_patch_id : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        """
        i_receiver_offset = 0
        for other_wall in self.other_wall_ids:
            if other_wall == receiver_wall_id:
                i_receiver_ff = receiver_patch_id
                break
            wall = patches_list[other_wall]
            i_receiver_offset += len(wall.patches)

        return self.form_factors[
            source_path_id, i_receiver_ff + i_receiver_offset]

    def energy_at_receiver(
            self, max_order, receiver,
            speed_of_sound, sampling_rate):
        """Calculate the energy at the receiver.

        this is supposed to be from just one wall

        Parameters
        ----------
        max_order : _type_
            _description_
        sound_source : _type_
            _description_
        receiver : _type_
            _description_
        speed_of_sound : float, optional
            _description_
        sampling_rate : int, optional
            _description_

        Returns
        -------
        _type_
            _description_

        """
        energy_response = np.zeros((self.n_bins, self.E_n_samples))

        for i_source, source_patch in enumerate(self.patches):
            source_pos = source_patch.center
            receiver_pos = receiver.position

            difference = np.abs(receiver_pos-source_pos)
            R = np.linalg.norm(source_pos-receiver_pos)
            delay = int(R/speed_of_sound * sampling_rate)

            cos_xi = np.abs(np.sum(source_patch.normal*difference)) / \
                np.linalg.norm(source_pos-receiver_pos)

            for k in range(max_order+1):
                for i_frequency in range(self.n_bins):
                    energy = self.E_matrix[i_frequency, k, i_source, :]
                    delayed_energy = _add_delay(energy, delay)

                    # Equation 20
                    factor = cos_xi * (np.exp(
                        -self.sound_attenuation_factor[i_frequency]*R)) / (
                            np.pi * R**2)
                    receiver_energy = delayed_energy * factor
                    if factor <0:
                        print(factor)

                    energy_response[i_frequency, ...] += receiver_energy

        return energy_response


class PatchesDirectionalKang(PatchesKang):
    """Class representing patches with directional scattering behaviour."""

    directivity_data: np.ndarray
    directivity_sources: pf.Coordinates
    directivity_receivers: pf.Coordinates

    def __init__(
            self, polygon, max_size, other_wall_ids, wall_id,
            data, sources, receivers, absorption=None,
            sound_attenuation_factor=None, already_converted=False,
            E_matrix=None):
        """Init Directional Patches.

        Parameters
        ----------
        polygon : Polygon
            Wall that should be distributed into patches.
        max_size : float
            maximal patchsize in meters.
        other_wall_ids : list[int]
            Ids of the other walls.
        wall_id : int
            id of this Patch
        data : pf.FrequencyData
            Directional Data
        sources : pf.Coordinates
            source positions of the directivity data
        receivers : pf.Coordinates
            receiver positions of the directivity data
        absorption : np.ndarray, optional
            Absorption coefficient of this wall for each freqeuncy
            bin, by default None
        sound_attenuation_factor : _type_, optional
            Air attenuation factor for each frequncy bin, by default None
        already_converted : bool, optional
            is ``sources`` and ``receivers`` already converted, by default
            False.
        E_matrix : np.ndarray, optional
            Energy exchange results, if already calcualted , by default None

        """
        self.directivity_data = data
        if absorption is None:
            absorption = np.zeros_like(data.frequencies)+0.1
        if sound_attenuation_factor is None:
            sound_attenuation_factor = np.zeros_like(data.frequencies)
        PatchesKang.__init__(
            self, polygon, max_size, other_wall_ids, wall_id,
            scattering=np.ones_like(data.frequencies),
            absorption=absorption,
            sound_attenuation_factor=sound_attenuation_factor,
            E_matrix=E_matrix,
        )
        self.directivity_data.freq = np.abs(self.directivity_data.freq)
        self.directivity_sources = sources
        self.directivity_receivers = receivers
        if not already_converted:
            self.directivity_data.freq *= np.pi
            o1 = pf.Orientations.from_view_up(
                polygon.normal, polygon.up_vector)
            o2 = pf.Orientations.from_view_up([0, 0, 1], [1, 0, 0])
            o_diff = o1.inv()*o2
            euler = o_diff.as_euler('xyz', True).flatten()
            self.directivity_receivers.rotate('xyz', euler)
            self.directivity_receivers.radius = 1
            self.directivity_sources.rotate('xyz', euler)
            self.directivity_sources.radius = 1

    @classmethod
    def from_sofa(cls, polygon, max_size, other_wall_ids, wall_id,
            wall_directivity_path, absorption=None,
            sound_attenuation_factor=None):
        """Create object with directional data from sofa."""
        sofa = sf.read_sofa(wall_directivity_path, True, False)
        data, sources, receivers = pf.io.convert_sofa(sofa)
        sources.weights = sofa.SourceWeights
        receivers.weights = sofa.ReceiverWeights
        assert sources.weights is not None, "No source weights in sofa file"
        assert receivers.weights is not None, \
            "No receiver weights in sofa file"
        return cls(
            polygon, max_size, other_wall_ids, wall_id,
            data, sources, receivers, absorption=absorption,
            sound_attenuation_factor=sound_attenuation_factor)

    def to_dict(self) -> dict:
        """Convert this object to dictionary. Used for read write."""
        return {
            **PatchesKang.to_dict(self),
            'directivity_data': self.directivity_data.freq,
            'directivity_data_frequencies': self.directivity_data.frequencies,
            'directivity_sources': self.directivity_sources.cartesian,
            'directivity_receivers': self.directivity_receivers.cartesian,
            'directivity_sources_weights': self.directivity_sources.weights,
            'directivity_receivers_weights': \
                self.directivity_receivers.weights,
        }

    @classmethod
    def from_dict(cls, input_dict: dict):
        """Create an object from a dictionary. Used for read write."""
        data = pf.FrequencyData(
            input_dict['directivity_data'],
            input_dict['directivity_data_frequencies'])
        sources = pf.Coordinates(
            np.array(input_dict['directivity_sources']).T[0],
            np.array(input_dict['directivity_sources']).T[1],
            np.array(input_dict['directivity_sources']).T[2],
            weights=input_dict['directivity_sources_weights'])
        receivers = pf.Coordinates(
            np.array(input_dict['directivity_receivers']).T[0],
            np.array(input_dict['directivity_receivers']).T[1],
            np.array(input_dict['directivity_receivers']).T[2],
            weights=input_dict['directivity_receivers_weights'])
        return cls(
            Polygon.from_dict(input_dict),
            max_size=input_dict['max_size'],
            other_wall_ids=input_dict['other_wall_ids'],
            wall_id=input_dict['wall_id'],
            data=data,
            sources=sources,
            receivers=receivers,
            absorption=input_dict['absorption'],
            sound_attenuation_factor=input_dict['sound_attenuation_factor'],
            E_matrix=input_dict['E_matrix'],
            already_converted=True,
            )

    def init_energy_exchange(
            self, max_order_k, ir_length_s, source,
            sampling_rate, speed_of_sound):
        """Initialize the energy exchange Matrix with source energy.

        Parameters
        ----------
        max_order_k : int
            max order of energy exchange iterations.
        ir_length_s : float
            length of the impulse response in seconds.
        source : SoundSource
            Sound source with ``sound_power`` and ``position``
        sampling_rate : int
            Sample rate of histogram, by default 1000 -> 1ms
        speed_of_sound : float,
            speed of sound in m/s.

        """
        PatchesKang.init_energy_exchange(
            self, max_order_k, ir_length_s, source, sampling_rate,
            speed_of_sound=speed_of_sound)
        test = self.E_matrix.copy()
        self.E_matrix = self.E_matrix[..., np.newaxis]
        self.E_matrix = np.tile(
            self.E_matrix, self.directivity_receivers.csize)
        assert np.sum(test-self.E_matrix[..., 0]) == 0
        patches_center = np.array([patch.center for patch in self.patches])
        difference = source.position - patches_center
        difference = pf.Coordinates(
            difference.T[0], difference.T[1], difference.T[2])
        source_idx = self.directivity_sources.find_nearest(difference)[0][0]
        # get directional scattering coefficient for
        # all incident angles x receiver
        for i_frequency in range(self.directivity_data.n_bins):
            scattering = self.directivity_data.freq[source_idx, :, i_frequency]
            if len(self.patches) == 1:
                self.E_matrix[i_frequency, 0, 0, :, :] *= np.abs(
                    scattering)
            else:
                for i_patch in range(len(self.patches)):
                    self.E_matrix[i_frequency, 0, i_patch, :, :] *= np.abs(
                        scattering[i_patch, :])

    def calculate_energy_exchange(
            self, patches_list, current_order_k, speed_of_sound,
            E_sampling_rate):
        """Calculate the energy exchange for a given order.

        It implements formula 18 and save the result in self.E_matrix.

        Parameters
        ----------
        patches_list : list[Patches]
            list of all patches
        current_order_k : int
            Order k
        speed_of_sound : int, optional
            speed of sound in m/s.
        E_sampling_rate : int, optional
            Sampling rate of histogram, e.g. 1000 Hz -> 1ms.

        """
        k = current_order_k # real k 1 .. max_k
        for i_receiver, receiver_patch in enumerate(self.patches):
            receiver = receiver_patch.center

            for wall_id in self.other_wall_ids:
                wall = patches_list[wall_id]
                for i_source, source_patch in enumerate(wall.patches):
                    source = source_patch.center

                    # distance between source and receiver patches
                    distance = np.linalg.norm(receiver-source)
                    delay_seconds = distance/speed_of_sound

                    # sample delay for the given distance
                    delay_samples = int(delay_seconds*E_sampling_rate)

                    # find form factor of source and receiver patches
                    form_factor = wall.get_form_factor(
                        patches_list, i_source, self.wall_id, i_receiver)

                    for i_frequency in range(self.directivity_data.n_bins):
                        # access energy matrix of source patch, previous order
                        A_k_minus_1 = wall.E_matrix[
                            i_frequency, k-1, i_source, :, :]

                        # delay IR by delay_samples
                        A_k_minus1_delay = _add_delay(
                            A_k_minus_1, delay_samples, 0)

                        # multiply delayed IR by form factor
                        alpha = self.absorption[i_frequency]

                        energy = A_k_minus1_delay * form_factor * (
                            1-alpha) * np.exp(
                                -self.sound_attenuation_factor[i_frequency] \
                                    * distance)

                        # find directional scattering coefficient
                        difference = source_patch.center - \
                            receiver_patch.center
                        difference = pf.Coordinates(
                            difference.T[0], difference.T[1], difference.T[2])
                        source_idx = self.directivity_sources.find_nearest(
                            difference)[0][0]
                        # get directional scattering coefficient for
                        # all incident angles x receiver
                        scattering = self.directivity_data.freq[
                            source_idx, :, i_frequency]

                        energy *= scattering

                        # add energy to energy matrix of self
                        self.E_matrix[
                            i_frequency, k, i_receiver, :, :] += energy

    def energy_at_receiver(
            self, max_order, receiver,
            speed_of_sound, sampling_rate):
        """Calculate the energy at the receiver.

        this is supposed to be from just one wall

        Parameters
        ----------
        max_order : int
            max order of energy exchange iterations.
        sound_source : SoundSource
            sound source with ``sound_power`` and ``position``
        receiver : Receiver
            receiver object with position.
        speed_of_sound : float, optional
            Speed of sound in m/s.
        sampling_rate : int, optional
            _description_, by default 1000

        Returns
        -------
        _type_
            _description_

        """
        energy_response = np.zeros((self.n_bins, self.E_n_samples))

        for i_source, source_patch in enumerate(self.patches):

            source_pos = source_patch.center
            receiver_pos = receiver.position

            cos_xi = np.abs(np.sum(
                source_patch.normal*np.abs(receiver_pos-source_pos))) / \
                np.linalg.norm(source_pos-receiver_pos)

            difference = receiver_pos-source_pos
            difference = pf.Coordinates(
                difference[0], difference[1], difference[2])
            difference.radius = 1
            R = np.linalg.norm(receiver_pos-source_pos)
            delay = int(R/speed_of_sound * sampling_rate)

            i_patch = self.directivity_receivers.find_nearest(
                difference)[0][0]

            for k in range(max_order+1):
                for i_frequency in range(self.n_bins):
                    energy = self.E_matrix[
                        i_frequency, k, i_source, :, i_patch]
                    delayed_energy = _add_delay(energy, delay, axis=-1)

                    # Equation 20
                    factor = cos_xi * (np.exp(
                        -self.sound_attenuation_factor[i_frequency]*R)) / (
                            np.pi * R**2)
                    if factor <0:
                        print(factor)
                    receiver_energy = delayed_energy * factor
                    energy_response[i_frequency, ...] += receiver_energy

        return energy_response


def _init_energy_exchange(
            dl, dm, dn, dd_l, dd_m, S_x, S_y, S_z,
            sound_power, absorption, distance, attenuation, n_bins):
    half_l = dd_l/2
    half_m = dd_m/2

    sin_phi_delta = (dl + half_l - S_x)/ (np.sqrt(np.square(
        dl+half_l-S_x) + np.square(dm-S_y) + np.square(dn-S_z)))

    test1 = (dl - half_l) <= S_x
    test2 = S_x <= (dl + half_l)
    k_phi = -1 if test1 and test2 else 1
    sin_phi = k_phi * (dl - half_l - S_x) / (np.sqrt(np.square(
        dl-half_l-S_x) + np.square(dm-S_y) + np.square(dn-S_z)))
    if (sin_phi_delta-sin_phi) < 1e-11:
        sin_phi *= -1

    plus  = np.arctan(np.abs((dm+half_m-S_y)/np.abs(S_z)))
    minus = np.arctan(np.abs((dm-half_m-S_y)/np.abs(S_z)))

    test1 = (dm - half_m) <= S_y
    test2 = S_y <= (dm + half_m)
    k_beta = -1 if test1 and test2 else 1

    beta = np.abs(plus-(k_beta*minus))

    # don't forget to add constants
    # constants are
    energies = np.zeros(n_bins)
    for i_frequency in range(n_bins):
        alpha = absorption[i_frequency]
        constant = sound_power * (1-alpha) * (
            np.exp(-attenuation[i_frequency]*distance))

        energy = constant * (
            np.abs(sin_phi_delta-sin_phi) ) * beta / (4*np.pi)
        energies[i_frequency] = energy
    return energies

def _add_delay(ir, delay_samples, axis=-1):
    """Add delay to impulse response.

    Parameters
    ----------
    ir : np.ndarray
        impulse response which should be shifted.
    delay_samples : int
        delay samples.
    axis : int, optional
        axis which should be rolled, by default -1

    Returns
    -------
    ir : np.ndarray
        shifted impulse response.

    """
    ## add delay
    if delay_samples > ir.shape[axis]:
        raise ValueError(
            'length of ir is longer then ir delay is '
            f'{delay_samples} > {ir.shape[-1]}')
    ir_delayed = np.roll(ir, delay_samples, axis=axis)
    return ir_delayed


def _calc_incident_direction(position, normal, up_vector, target_position):
    """Calculate the incident direction of a sound wave.

    Parameters
    ----------
    position : np.ndarray
        Position of the wall.
    normal : np.ndarray
        Normal vector of the wall.
    up_vector : np.ndarray
        Up vector of the wall.
    target_position : np.ndarray
        Position of the source or other wall.

    Returns
    -------
    pf.Coordinates
        Incident direction of the sound wave.

    """
    direction = np.array(target_position) - np.array(position)

    x_dash = np.cross(normal, up_vector)
    y_dash = np.array(up_vector)
    z_dash = -np.array(normal)

    w_x_local = np.dot(direction, x_dash)
    w_y_local = np.dot(direction, y_dash)
    w_z_local = np.dot(direction, z_dash)

    azimuth = np.arctan2(-w_x_local, -w_z_local)
    w_y_local_normalized = w_y_local / np.linalg.norm(
        [w_x_local, w_y_local, w_z_local])
    elevation = np.arcsin(w_y_local_normalized)

    return pf.Coordinates.from_spherical_colatitude(
        azimuth, elevation, 1)


class RadiosityKang():
    """Radiosity object simulation."""

    speed_of_sound: float
    max_order_k: int
    sampling_rate: int
    patch_size: float
    ir_length_s: float
    patch_list: list[PatchesKang]

    def __init__(
            self, walls, patch_size, max_order_k, ir_length_s,
            speed_of_sound=346.18, sampling_rate=1000, absorption=0.1,
            source=None):
        """Create Radiosity Object for simulation.

        Parameters
        ----------
        walls : list[Polygon]
            list of patches
        patch_size : float
            maximal patchsize in meters.
        max_order_k : int
            max order of energy exchange iterations.
        ir_length_s : float
            length of ir in seconds.
        speed_of_sound : float, optional
            Speed of sound in m/s, by default 346.18 m/s
        sampling_rate : int, optional
            sampling rate of the Energy histogram, by default 1000 -> 1ms
        absorption: np.ndarray
            Absorption coefficient of this wall for each frequency bin
        source : SoundSource, optional
            Source object, by default None, can be added later.

        """
        self.speed_of_sound = speed_of_sound
        self.max_order_k = max_order_k
        self.sampling_rate = sampling_rate
        self.patch_size = patch_size
        self.ir_length_s = ir_length_s

        # A. Patch division
        patch_list = []
        n_patches = len(walls)
        for i, wall in enumerate(walls):
            index_list = [j for j in range(n_patches) if j != i]
            patches = wall if isinstance(wall, PatchesKang) else PatchesKang(
                wall, self.patch_size, index_list, i,absorption=absorption)
            patch_list.append(patches)

        self.patch_list = patch_list
        if source is not None:
            self.source = source

    def to_dict(self) -> dict:
        """Convert this object to dictionary. Used for read write."""
        is_source = hasattr(self, 'source')
        source = self.source if is_source else None
        return {
            'patch_size': self.patch_size,
            'max_order_k': self.max_order_k,
            'ir_length_s': self.ir_length_s,
            'speed_of_sound': self.speed_of_sound,
            'sampling_rate': self.sampling_rate,
            'patches': [patch.to_dict() for patch in self.patch_list],
            'source_position': source.position if is_source else None,
            'source_view': source.view if is_source else None,
            'source_up': source.up if is_source else None,
        }

    @classmethod
    def from_dict(cls, input_dict: dict):
        """Create an object from a dictionary. Used for read write."""
        patch_list = [
            PatchesKang.from_dict(patch) for patch in input_dict['patches']]
        source = None
        if input_dict is not None:
            source = SoundSource(
                input_dict['source_position'],
                input_dict['source_view'], input_dict['source_up'])
        obj = cls(
            patch_list, input_dict['patch_size'], input_dict['max_order_k'],
            input_dict['ir_length_s'],
            input_dict['speed_of_sound'], input_dict['sampling_rate'],
            source=source)
        return obj

    def run(self, source):
        """Execute the radiosity algorithm."""
        self.source = source
        # B. First-order patch sources
        for patches in self.patch_list:
            patches.init_energy_exchange(
                self.max_order_k, self.ir_length_s, source,
                sampling_rate=self.sampling_rate,
                speed_of_sound=self.speed_of_sound)

        # C. Form factors
        if len(self.patch_list) > 1:
            for patches in self.patch_list:
                patches.calculate_form_factor(self.patch_list)

        # D. Energy exchange between patches
        if len(self.patch_list) > 1:
            for k in range(1, self.max_order_k+1):
                for patches in self.patch_list:
                    patches.calculate_energy_exchange(
                        self.patch_list, k, speed_of_sound=self.speed_of_sound,
                        E_sampling_rate=self.sampling_rate)

    def energy_at_receiver(
            self, receiver, max_order_k=None, ignore_direct=False):
        """Return the energetic impulse response at the receiver."""
        ir = 0
        if max_order_k is None:
            max_order_k = self.max_order_k
        M_value = self.patch_list[0].sound_attenuation_factor
        for patches in self.patch_list:
            ir += patches.energy_at_receiver(
                max_order_k, receiver,
                speed_of_sound=self.speed_of_sound,
                sampling_rate=self.sampling_rate)
        if not ignore_direct:
            r = np.sqrt(np.sum((receiver.position-self.source.position)**2))
            direct_sound = (1/(4 * np.pi * np.square(r))) * np.exp(-M_value*r)
            delay_dir = int(r/self.speed_of_sound*self.sampling_rate)
            ir[:, delay_dir] += direct_sound

        return ir

    def write(self, filename, compress=True):
        """Write the object to a far file."""
        pf.io.write(filename, compress=compress, **self.to_dict())

    @classmethod
    def from_read(cls, filename):
        """Read the object to a far file."""
        data = pf.io.read(filename)
        return cls.from_dict(data)


class DirectionalRadiosityKang():
    """Radiosity object for directional scattering coefficients."""

    def __init__(
            self, polygon_list, patch_size, max_order_k, ir_length_s,
            sofa_path, speed_of_sound=346.18, sampling_rate=1000, source=None):
        """Create a Radiosity object for directional scattering coefficients.

        Parameters
        ----------
        polygon_list : list[PatchesDirectional]
            list of patches
        patch_size : float
            maximal patchsize in meters.
        max_order_k : int
            max order of energy exchange iterations.
        ir_length_s : float
            length of ir in seconds.
        sofa_path : Path, string, list of Path, list of string
            path of directional scattering coefficients or list of path
            for each Patch.
        speed_of_sound : float, optional
            Speed of sound in m/s, by default 346.18 m/s
        sampling_rate : int, optional
            sampling rate of the Energy histogram, by default 1000 -> 1ms
        source : SoundSource, optional
            Source object, by default None, can be added later.

        """
        self.speed_of_sound = speed_of_sound
        self.max_order_k = max_order_k
        self.sampling_rate = sampling_rate
        self.patch_size = patch_size
        self.ir_length_s = ir_length_s

        # A. Patch division
        patch_list = []
        n_points = len(polygon_list)
        for i in range(n_points):
            index_list = [j for j in range(n_points) if j != i]
            path = sofa_path[i] if isinstance(sofa_path, list) else sofa_path
            patches = polygon_list[i] if isinstance(
                polygon_list[i], PatchesDirectionalKang) else \
                    PatchesDirectionalKang.from_sofa(
                    polygon_list[i], self.patch_size, index_list, i, path)
            patch_list.append(patches)

        self.patch_list = patch_list
        n_bins = self.patch_list[0].n_bins
        for patches in self.patch_list:
            assert patches.n_bins == n_bins, \
                'Number of bins is not the same for all patches. ' + \
                f'{patches.n_bins} != {n_bins}'

        if source is not None:
            self.source = source

    def write(self, filename, compress=True):
        """Write the object to a far file."""
        pf.io.write(filename, compress=compress, **self.to_dict())

    @classmethod
    def from_read(cls, filename):
        """Read the object to a far file."""
        data = pf.io.read(filename)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Convert this object to dictionary. Used for read write."""
        is_source = hasattr(self, 'source')
        source = self.source if is_source else None
        return {
            'patch_size': self.patch_size,
            'max_order_k': self.max_order_k,
            'ir_length_s': self.ir_length_s,
            'speed_of_sound': self.speed_of_sound,
            'sampling_rate': self.sampling_rate,
            'patches': [patch.to_dict() for patch in self.patch_list],
            'source_position': source.position if is_source else None,
            'source_view': source.view if is_source else None,
            'source_up': source.up if is_source else None,
        }

    @classmethod
    def from_dict(cls, input_dict: dict):
        """Create an object from a dictionary. Used for read write."""
        patch_list = [
            PatchesDirectionalKang.from_dict(patch) for patch in input_dict[
                'patches']]
        source = None
        if input_dict['source_position'] is not None:
            source = SoundSource(
                input_dict['source_position'],
                input_dict['source_view'], input_dict['source_up'])
        obj = cls(
            patch_list, input_dict['patch_size'], input_dict['max_order_k'],
            input_dict['ir_length_s'],
            sofa_path=None,
            speed_of_sound=input_dict['speed_of_sound'],
            sampling_rate=input_dict['sampling_rate'],
            source=source,
            )
        return obj

    def run(self, source):
        """Execute the radiosity algorithm."""
        self.source = source
        # B. First-order patch sources
        for patches in self.patch_list:
            patches.init_energy_exchange(
                self.max_order_k, self.ir_length_s, source,
                sampling_rate=self.sampling_rate,
                speed_of_sound=self.speed_of_sound)

        # C. Form factors
        if len(self.patch_list) > 1:
            for patches in self.patch_list:
                patches.calculate_form_factor(self.patch_list)

        # D. Energy exchange between patches
        if len(self.patch_list) > 1:
            for k in range(1, self.max_order_k+1):
                for patches in self.patch_list:
                    patches.calculate_energy_exchange(
                        self.patch_list, k, speed_of_sound=self.speed_of_sound,
                        E_sampling_rate=self.sampling_rate)

    def energy_at_receiver(self, receiver, order_k=None):
        """Return the energetic impulse response at the receiver."""
        ir = 0
        if order_k is None:
            order_k = self.max_order_k
        M = self.patch_list[0].sound_attenuation_factor
        for patches in self.patch_list:
            ir += patches.energy_at_receiver(
                order_k, receiver,
                speed_of_sound=self.speed_of_sound,
                sampling_rate=self.sampling_rate)
        r = np.sqrt(np.sum((receiver.position-self.source.position)**2))
        direct_sound = (1/(4 * np.pi * np.square(r))) * np.exp(-M*r)
        delay_dir = int(r/self.speed_of_sound*self.sampling_rate)
        ir[:, delay_dir] += direct_sound

        return ir
