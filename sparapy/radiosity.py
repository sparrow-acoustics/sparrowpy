"""Module for the radiosity simulation."""
import matplotlib as mpl
import numpy as np
import pyfar as pf
import sofar as sf
from tqdm import tqdm

from sparapy.geometry import Polygon, SoundSource


class Patches(Polygon):
    """Class representing patches of a polygon."""

    patches: list[Polygon]
    other_wall_ids: list[int]  # ids of the patches
    form_factors: np.ndarray
    E_matrix: np.ndarray
    wall_id : int
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
        min = np.min(polygon.pts, axis=0)
        max = np.max(polygon.pts, axis=0)
        size = max - min
        patch_nums = [int(n) for n in size/max_size]
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
    def from_dict(cls, dict: dict):
        """Create an object from a dictionary. Used for read write."""
        return cls(
            Polygon.from_dict(dict),
            dict['max_size'], dict['other_wall_ids'], dict['wall_id'],
            dict['scattering'], dict['absorption'],
            dict['sound_attenuation_factor'],
            E_matrix=dict['E_matrix'])

    def plot(self, ax: mpl.axes.Axes = None, color=None):
        """Plot the patches."""
        for patch in self.patches:
            patch.plot_point(ax, color)
        self.plot_view_up(ax, color)

    def init_energy_exchange(
            self, max_order_k, ir_length_s, source, sampling_rate=1000,
            speed_of_sound=346.18):
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
            Sampling rate of impulse response, by default 1000 Hz.
        speed_of_sound : float, optional
            speed of sound in m/s, by default 346.18 m/s.

        """
        self.E_sampling_rate = sampling_rate
        self.E_n_samples = int(ir_length_s*sampling_rate)

        self.E_matrix = np.zeros((
            self.n_bins,
            max_order_k+1,  # order of energy exchange
            len(self.patches),  # receiver ids of patches
            self.E_n_samples,  # impulse response G_k(t)_receiverpatch
            ))

        S_x = source.position[0]
        S_y = source.position[1]
        S_z = source.position[2]

        source_pos = source.position

        for i_receiver, receiver_patch in enumerate(self.patches):
            receiver_pos = receiver_patch.center

            distance = np.linalg.norm(receiver_pos-source_pos)
            delay_seconds = distance/speed_of_sound
            delay_samples = int(delay_seconds*self.E_sampling_rate)

            if np.abs(receiver_patch.normal[2]) > 0.99:
                dl = receiver_patch.center[0]
                dm = receiver_patch.center[1]
                dn = receiver_patch.center[2]
                dd_l = receiver_patch.size[0]
                dd_m = receiver_patch.size[1]
                dd_n = receiver_patch.size[2]
                S_x = source.position[0]
                S_y = source.position[1]
                S_z = source.position[2]
            elif np.abs(receiver_patch.normal[1]) > 0.99:
                dl = receiver_patch.center[0]
                dm = receiver_patch.center[2]
                dn = receiver_patch.center[1]
                dd_l = receiver_patch.size[0]
                dd_m = receiver_patch.size[2]
                dd_n = receiver_patch.size[1]
                S_x = source.position[0]
                S_y = source.position[2]
                S_z = source.position[1]
            elif np.abs(receiver_patch.normal[0]) > 0.99:
                dl = receiver_patch.center[1]
                dm = receiver_patch.center[2]
                dn = receiver_patch.center[0]
                dd_l = receiver_patch.size[1]
                dd_m = receiver_patch.size[2]
                dd_n = receiver_patch.size[0]
                S_x = source.position[1]
                S_y = source.position[2]
                S_z = source.position[0]
            else:
                raise AssertionError()

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

            # don't forget to add constants
            # constants are
            for i_frequency in range(self.n_bins):
                alpha = self.absorption[i_frequency]
                constant = source.sound_power * (1-alpha) * (
                    np.exp(-self.sound_attenuation_factor[i_frequency]*distance))
                #constant = 1
                energy = constant * (
                    np.abs(sin_phi_delta-sin_phi) ) * beta / (4*np.pi)
                # energy = round(energy,4)
                self.E_matrix[i_frequency, 0, i_receiver, delay_samples] += energy

    def calculate_energy_exchange(
            self, patches_list, current_order_k, speed_of_sound=346.18,
            E_sampling_rate=1000):
        """Calculate the energy exchange for a given order.

        It implements formula 18 and save the result in self.E_matrix.

        Parameters
        ----------
        patches_list : list[Patches]
            list of all patches
        current_order_k : int
            Order k
        speed_of_sound : int, optional
            speed of sound in m/s, by default 346.18 m/s.
        E_sampling_rate : int, optional
            Sampling rate of histogram, by default 1000 Hz -> 1ms.

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
                        A_k_minus1_delay = add_delay(A_k_minus_1, delay_samples)

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
                        # energy[3]
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
                            idx_source = set([2, 1])
                            dl = source_center[2]
                            dm = source_center[1]
                        elif np.abs(source_patch.normal[1]) > 1e-5:
                            idx_source = set([2, 0])
                            dl = source_center[2]
                            dm = source_center[0]
                        elif np.abs(source_patch.normal[2]) > 1e-5:
                            idx_source = set([0, 1])
                            dl = source_center[1]
                            dm = source_center[0]

                        if np.abs(receiver_patch.normal[0]) > 1e-5:
                            idx_receiver = set([2, 1])
                            dl_prime = receiver_center[1]
                            dn_prime = receiver_center[2]
                        elif np.abs(receiver_patch.normal[1]) > 1e-5:
                            idx_receiver = set([0, 2])
                            dl_prime = receiver_center[0]
                            dn_prime = receiver_center[2]
                        elif np.abs(receiver_patch.normal[2]) > 1e-5:
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

                        # if source_patch.center[2] == 0:
                        #     # source is ground, receiver is wall --> UNSURE
                        #     assert dl == source_patch.center[0]
                        #     assert dm == source_patch.center[1]
                        #     assert dl_prime == receiver_patch.center[0]
                        #     assert dn_prime == receiver_patch.center[2]

                        # else:
                        #     # source is wall, receiver is ground
                        #     assert dl == receiver_patch.center[0]
                        #     assert dm == receiver_patch.center[1]
                        #     assert dl_prime == source_patch.center[0]
                        #     assert dn_prime == source_patch.center[2]

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
        for idx, other_wall in enumerate(self.other_wall_ids):
            if other_wall == receiver_wall_id:
                i_receiver_ff = receiver_patch_id
                break
            wall = patches_list[other_wall]
            #wall = patches_list[idx]
            i_receiver_offset += len(wall.patches)

        return self.form_factors[
            source_path_id, i_receiver_ff + i_receiver_offset]

    def energy_at_receiver(
            self, max_order, receiver, ir_length_s,
            speed_of_sound=346.18, sampling_rate=1000):
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
        ir_length_s : _type_
            _description_
        speed_of_sound : float, optional
            _description_, by default 346.18
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

            R = np.linalg.norm(source_pos-receiver_pos)
            delay = int(R/speed_of_sound * sampling_rate)

            cos_xi = np.abs(np.sum(source_patch.normal*receiver_pos)) / \
                np.linalg.norm(source_patch.center-receiver_pos)

            for k in range(max_order+1):
                for i_frequency in range(self.n_bins):
                    energy = self.E_matrix[i_frequency, k, i_source, :]
                    delayed_energy = add_delay(energy, delay)

                    # Equation 20
                    factor = cos_xi * (np.exp(
                        -self.sound_attenuation_factor[i_frequency]*R)) / (np.pi * R**2)
                    receiver_energy = delayed_energy * factor
                    if factor <0:
                        print(factor)

                    energy_response[i_frequency, ...] += receiver_energy

        return energy_response


class PatchesDirectional(Patches):
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
            is ``sources`` and ``receivers`` already converted, by default False
        E_matrix : np.ndarray, optional
            Energy exchange results, if already calcualted , by default None

        """
        self.directivity_data = data
        if absorption is None:
            absorption = np.zeros_like(data.frequencies)+0.1
        if sound_attenuation_factor is None:
            sound_attenuation_factor = np.zeros_like(data.frequencies)
        Patches.__init__(
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
            o1 = pf.Orientations.from_view_up(
                polygon.normal, polygon.up_vector)
            o2 = pf.Orientations.from_view_up([0, 0, 1], [1, 0, 0])
            o_diff = o1.inv()*o2
            euler = o_diff.as_euler('xyz', True).flatten()
            self.directivity_receivers.rotate('xyz', euler)
            self.directivity_receivers.radius = 1
            self.directivity_sources.rotate('xyz', euler)
            self.directivity_sources.radius = 1
            # to make same with just ones in diffsue case
            self.directivity_data.freq *= receivers.csize

    @classmethod
    def from_sofa(cls, polygon, max_size, other_wall_ids, wall_id,
            wall_directivity_path, absorption=None,
            sound_attenuation_factor=None):
        """Create object with directional data from sofa."""
        sofa = sf.read_sofa(wall_directivity_path, True, False)
        data, sources, receivers = pf.io.convert_sofa(sofa)
        return cls(
            polygon, max_size, other_wall_ids, wall_id,
            data, sources, receivers, absorption=absorption,
            sound_attenuation_factor=sound_attenuation_factor)

    def to_dict(self) -> dict:
        """Convert this object to dictionary. Used for read write."""
        return {
            **Patches.to_dict(self),
            'directivity_data': self.directivity_data.freq,
            'directivity_data_frequencies': self.directivity_data.frequencies,
            'directivity_sources': self.directivity_sources.cartesian,
            'directivity_receivers': self.directivity_receivers.cartesian,
        }

    @classmethod
    def from_dict(cls, dict: dict):
        """Create an object from a dictionary. Used for read write."""
        data = pf.FrequencyData(
            dict['directivity_data'], dict['directivity_data_frequencies'])
        sources = pf.Coordinates(
            np.array(dict['directivity_sources']).T[0],
            np.array(dict['directivity_sources']).T[1],
            np.array(dict['directivity_sources']).T[2])
        receivers = pf.Coordinates(
            np.array(dict['directivity_receivers']).T[0],
            np.array(dict['directivity_receivers']).T[1],
            np.array(dict['directivity_receivers']).T[2])
        return cls(
            Polygon.from_dict(dict),
            max_size=dict['max_size'],
            other_wall_ids=dict['other_wall_ids'],
            wall_id=dict['wall_id'],
            data=data,
            sources=sources,
            receivers=receivers,
            absorption=dict['absorption'],
            sound_attenuation_factor=dict['sound_attenuation_factor'],
            E_matrix=dict['E_matrix'],
            already_converted=True,
            )

    def init_energy_exchange(
            self, max_order_k, ir_length_s, source,
            sampling_rate=1000):
        """Initialize the energy exchange Matrix with source energy.

        Parameters
        ----------
        max_order_k : int
            max order of energy exchange iterations.
        ir_length_s : float
            length of the impulse response in seconds.
        source : SoundSource
            Sound source with ``sound_power`` and ``position``
        sampling_rate : int, optional
            Sample rate of histogram, by default 1000 -> 1ms

        """
        Patches.init_energy_exchange(
            self, max_order_k, ir_length_s, source, sampling_rate)
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
            # assert all(np.sum(np.real(scattering), axis=-1)-1 < 1e-5)
            if len(self.patches) == 1:
                self.E_matrix[i_frequency, 0, 0, :, :] *= np.abs(scattering)
            else:
                for i_patch in range(len(self.patches)):
                    self.E_matrix[i_frequency, 0, i_patch, :, :] *= np.abs(
                        scattering[i_patch, :])

    def calculate_energy_exchange(
            self, patches_list, current_order_k, speed_of_sound=346.18,
            E_sampling_rate=1000):
        """Calculate the energy exchange for a given order.

        It implements formula 18 and save the result in self.E_matrix.

        Parameters
        ----------
        patches_list : list[Patches]
            list of all patches
        current_order_k : int
            Order k
        speed_of_sound : int, optional
            speed of sound in m/s, by default 346.18 m/s.
        E_sampling_rate : int, optional
            Sampling rate of histogram, by default 1000 Hz -> 1ms.

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
                        A_k_minus1_delay = add_delay(
                            A_k_minus_1, delay_samples, 0)

                        # multiply delayed IR by form factor
                        alpha = self.absorption[i_frequency]

                        energy = A_k_minus1_delay * form_factor * (
                            1-alpha) * np.exp(
                                -self.sound_attenuation_factor[i_frequency] \
                                    * distance)

                        # find directional scattering coefficient
                        difference = source_patch.center - receiver_patch.center
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
                        self.E_matrix[i_frequency, k, i_receiver, :, :] += energy
                        # energy[3]
                        # 0.00120904

    def energy_at_receiver(
            self, max_order, receiver, ir_length_s,
            speed_of_sound=346.18, sampling_rate=1000, M=0):
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
        ir_length_s : float
            useless
        speed_of_sound : float, optional
            Speed of sound in m/s, by default 346.18 m/s.
        sampling_rate : int, optional
            _description_, by default 1000
        M : float, optional
            Air attenuation factor in Np/m, by default 0.

        Returns
        -------
        _type_
            _description_

        """
        energy_response = np.zeros((self.n_bins, self.E_n_samples))

        for i_source, source_patch in enumerate(self.patches):
            source_pos = source_patch.center
            receiver_pos = receiver.position

            difference = receiver_pos-source_pos
            difference = pf.Coordinates(
                difference[0], difference[1], difference[2])
            difference.radius = 1
            R = np.linalg.norm(receiver_pos-source_pos)
            delay = int(R/speed_of_sound * sampling_rate)

            cos_xi = np.abs(np.sum(source_patch.normal*receiver_pos)) / \
                np.linalg.norm(source_pos-receiver_pos)

            i_patch = self.directivity_receivers.find_nearest(
                difference)[0][0]

            for k in range(max_order+1):
                for i_frequency in range(self.n_bins):
                    energy = self.E_matrix[i_frequency, k, i_source, :, i_patch]
                    delayed_energy = add_delay(energy, delay, axis=-1)

                    # Equation 20
                    factor = cos_xi * (np.exp(
                        -self.sound_attenuation_factor[i_frequency]*R)) / (
                            np.pi * R**2)
                    if factor <0:
                        print(factor)
                    receiver_energy = delayed_energy * factor
                    energy_response[i_frequency, ...] += receiver_energy

        return energy_response

def add_delay(ir, delay_samples, axis=-1):
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
            f'length of ir is longer then ir delay is {delay_samples} > {ir.shape[-1]}')
    ir_delayed = np.roll(ir, delay_samples, axis=axis)
    # if np.sum(ir_delayed[:delay_samples, ...]) != 0:
    #     raise ValueError('length of ir is to short, ')
    return ir_delayed


def calc_incident_direction(position, normal, up_vector, target_position):
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
    # distance = np.linalg.norm(direction)

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


class Radiosity():
    """Radiosity object simulation."""

    speed_of_sound: float
    max_order_k: int
    sampling_rate: int
    patch_size: float
    ir_length_s: float
    # source: SoundSource
    patch_list: list[Patches]

    def __init__(
            self, walls, patch_size, max_order_k, ir_length_s,
            speed_of_sound=346.18, sampling_rate=1000, source=None):
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
            samplingrate of the Energy histogram, by default 1000 -> 1ms
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
            patches = wall if isinstance(wall, Patches) else Patches(
                wall, self.patch_size, index_list, i)
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
    def from_dict(cls, dict: dict):
        """Create an object from a dictionary. Used for read write."""
        patch_list = [
            Patches.from_dict(patch) for patch in dict['patches']]
        source = None
        if dict is not None:
            source = SoundSource(
                dict['source_position'],
                dict['source_view'], dict['source_up'])
        obj = cls(
            patch_list, dict['patch_size'], dict['max_order_k'], dict['ir_length_s'],
            dict['speed_of_sound'], dict['sampling_rate'], source)
        return obj

    def run(self, source):
        """Execute the radiosity algorithm."""
        self.source = source
        # B. First-order patch sources
        for patches in self.patch_list:
            patches.init_energy_exchange(
                self.max_order_k, self.ir_length_s, source,
                sampling_rate=self.sampling_rate)

        # C. Form factors
        if len(self.patch_list) > 1:
            for patches in self.patch_list:
                patches.calculate_form_factor(self.patch_list)

        # D. Energy exchange between patches
        if len(self.patch_list) > 1:
            for k in tqdm(range(1, self.max_order_k+1)):
                for patches in self.patch_list:
                    patches.calculate_energy_exchange(
                        self.patch_list, k, speed_of_sound=self.speed_of_sound,
                        E_sampling_rate=self.sampling_rate)

    def energy_at_receiver(self, receiver, max_order_k=None):
        """Return the energetic impulse response at the receiver."""
        ir = 0
        if max_order_k is None:
            max_order_k = self.max_order_k
        M_value = self.patch_list[0].sound_attenuation_factor
        for patches in self.patch_list:
            ir += patches.energy_at_receiver(
                max_order_k, receiver, self.ir_length_s,
                speed_of_sound=self.speed_of_sound,
                sampling_rate=self.sampling_rate)
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


class DirectionalRadiosity():
    """Radiosity object for directional scattering coefficients."""

    def __init__(
            self, polygon_list, patch_size, max_order_k, ir_length_s,
            sofa_path, speed_of_sound=346.18, sampling_rate=1000, source=None):
        """Create a Radiosoty object for directional scattering coefficents.

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
            samplingrate of the Energy histogram, by default 1000 -> 1ms
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
                polygon_list[i], PatchesDirectional) else PatchesDirectional.from_sofa(
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
    def from_dict(cls, dict: dict):
        """Create an object from a dictionary. Used for read write."""
        patch_list = [
            PatchesDirectional.from_dict(patch) for patch in dict['patches']]
        source = None
        if dict['source_position'] is not None:
            source = SoundSource(
                dict['source_position'],
                dict['source_view'], dict['source_up'])
        obj = cls(
            patch_list, dict['patch_size'], dict['max_order_k'],
            dict['ir_length_s'],
            sofa_path=None,
            speed_of_sound=dict['speed_of_sound'],
            sampling_rate=dict['sampling_rate'],
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
                sampling_rate=self.sampling_rate)

        # C. Form factors
        if len(self.patch_list) > 1:
            for patches in self.patch_list:
                patches.calculate_form_factor(self.patch_list)

        # D. Energy exchange between patches
        if len(self.patch_list) > 1:
            for k in tqdm(range(1, self.max_order_k+1)):
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
                order_k, receiver, self.ir_length_s,
                speed_of_sound=self.speed_of_sound,
                sampling_rate=self.sampling_rate)
        r = np.sqrt(np.sum((receiver.position-self.source.position)**2))
        direct_sound = (1/(4 * np.pi * np.square(r))) * np.exp(-M*r)
        delay_dir = int(r/self.speed_of_sound*self.sampling_rate)
        ir[:, delay_dir] += direct_sound

        return ir
