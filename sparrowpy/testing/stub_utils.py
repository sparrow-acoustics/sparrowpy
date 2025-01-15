"""Stub utilities for testing."""
import sparrowpy as sp
import numpy as np
import pyfar as pf


def shoebox_room_stub(length_x, length_y, length_z):
    """Create a shoebox room with the given dimensions.

    Parameters
    ----------
    length_x : float
        Length of the room in x-direction in meters.
    length_y : float
        Length of the room in y-direction in meters
    length_z : float
        Length of the room in z-direction in meters

    Returns
    -------
    room : list[geo.Polygon]
        List of the walls of the room.

    """
    return [
        sp.geometry.Polygon(
            [[0, 0, 0], [length_x, 0, 0],
            [length_x, 0, length_z], [0, 0, length_z]],
            [1, 0, 0], [0, 1, 0]),
        sp.geometry.Polygon(
            [[0, length_y, 0], [length_x, length_y, 0],
            [length_x, length_y, length_z], [0, length_y, length_z]],
            [1, 0, 0], [0, -1, 0]),
        sp.geometry.Polygon(
            [[0, 0, 0], [length_x, 0, 0],
            [length_x, length_y, 0], [0, length_y, 0]],
            [1, 0, 0], [0, 0, 1]),
        sp.geometry.Polygon(
            [[0, 0, length_z], [length_x, 0, length_z],
            [length_x, length_y, length_z], [0, length_y, length_z]],
            [1, 0, 0], [0, 0, -1]),
        sp.geometry.Polygon(
            [[0, 0, 0], [0, 0, length_z],
            [0, length_y, length_z], [0, length_y, 0]],
            [0, 0, 1], [1, 0, 0]),
        sp.geometry.Polygon(
            [[length_x, 0, 0], [length_x, 0, length_z],
            [length_x, length_y, length_z], [length_x, length_y, 0]],
            [0, 0, 1], [-1, 0, 0]),
        ]


def infinite_plane(
        source:np.ndarray,
        receiver:np.ndarray,
        ratio=30.):
    """Create an "infinite" xy plane relative to eval point position.

    Parameters
    ----------
    source : np.array(3,)
        source position in 3D space
    receiver : np.array(3,)
        receiver position in 3D space
    ratio: float
        ratio between plane dimensions and
        distance between receiver and plane.

    Returns
    -------
    plane: sparrowpy.geometry.Polygon
        plane surface in format compatible with sparrowpy.

    """
    length = ratio*max(source[2],receiver[2]) # dimensions of plane's side

    center_x = source[0]-(source[0]-receiver[0])/2
    center_y = source[1]-(source[1]-receiver[1])/2

    w = (abs(source[0]-receiver[0])+length)/2
    h = (abs(source[1]-receiver[1])+length)/2

    return  [
            sp.geometry.Polygon([ [center_x+w, center_y+h, 0],
                                  [center_x-w, center_y+h, 0],
                                  [center_x-w, center_y-h, 0],
                                  [center_x+w, center_y-h, 0],
                                ],
                                up_vector=np.array([1.,0.,0.]),
                                normal=np.array([0.,0.,1.])),
            ]


def get_histogram(source_pos=np.array([0.,0.,1.]),
                  receiver_pos=np.array([0.,0.,1.]),
                  sampling_rate=100.,
                  h2ps_ratio=1.,
                  frequencies=np.array([1000.]),
                  ):
    """Generate histogram of infinite plane scenario.

    Parameters
    ----------
    source_pos: np.ndarray((3,), dtype=float)
        source position in space.
        *z>0!*
    receiver_pos: np.ndarray((3,), dtype=float)
        receiver position in space.
        *z>0!*
    sampling_rate: int
        sampling rate of histogram in Hz.
    h2ps_ratio: float
        ratio of patch size relative to maximum
        source/receiver height.
    frequencies : np.ndarray
        Frequency bins in Hz.

    """
    ## BASIC STUFF ##
    #generate "infinite" plane
    plane = infinite_plane(receiver=receiver_pos,source=source_pos)

    #determine patch size based on input ratio
    patch_size = h2ps_ratio*max(receiver_pos[2],source_pos[2])

    #simulation parameters
    speed_of_sound = 346.18
    max_sound_path_length = np.sqrt(
        (plane.pts[0]-plane.pts[2])[0]**2 +
        (plane.pts[0]-plane.pts[2])[1]**2 +
        max(receiver_pos[2],source_pos[2])**2
    )
    max_histogram_length = max_sound_path_length/speed_of_sound

    ## PREPARE RADIOSITY SIMULATION ##
    #initialize radiosity class instance based on plane "radi"
    radi = sp.DRadiosityFast.from_polygon(plane,patch_size)

    #set scattering distribution (lambertian surface)
    scattering_data = pf.FrequencyData(
                np.ones((1, 1, frequencies.size)), frequencies)
    radi.set_wall_scattering(np.arange(1), scattering_data,
                             sources=np.array([source_pos]),
                             receivers=np.array([receiver_pos]))

    #set atmospheric attenuation and surface absorption coefficient to 0
    radi.set_air_attenuation(
                    pf.FrequencyData(
                        np.zeros_like(scattering_data.frequencies),
                        scattering_data.frequencies))
    radi.set_wall_absorption(
                    np.arange(1),
                    pf.FrequencyData(
                        np.zeros_like(scattering_data.frequencies),
                        scattering_data.frequencies))

    ## RUN SIMULATION ##
    #precompute energy relationships
    radi.bake_geometry()

    #initialize source energy (source cast)
    radi.init_source_energy(source_position=source_pos)

    #energy propagation simulation
    radi.calculate_energy_exchange(
        speed_of_sound=speed_of_sound,
        histogram_time_resolution=1/sampling_rate,
        histogram_length=1.*max_histogram_length,
    )
