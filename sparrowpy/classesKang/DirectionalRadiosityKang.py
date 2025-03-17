"""Module for the radiosity simulation."""
import numpy as np
import pyfar as pf
import sofar as sf
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        """Dummy tqdm function."""
        return x
from sparrowpy.geometry import Polygon, SoundSource
from .RadiosityKang import Patches


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
            **Patches.to_dict(self),
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
        Patches.init_energy_exchange(
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


class DirectionalRadiosityKang():
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
                polygon_list[i], PatchesDirectional) else \
                    PatchesDirectional.from_sofa(
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
            PatchesDirectional.from_dict(patch) for patch in input_dict[
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
                order_k, receiver,
                speed_of_sound=self.speed_of_sound,
                sampling_rate=self.sampling_rate)
        r = np.sqrt(np.sum((receiver.position-self.source.position)**2))
        direct_sound = (1/(4 * np.pi * np.square(r))) * np.exp(-M*r)
        delay_dir = int(r/self.speed_of_sound*self.sampling_rate)
        ir[:, delay_dir] += direct_sound

        return ir
