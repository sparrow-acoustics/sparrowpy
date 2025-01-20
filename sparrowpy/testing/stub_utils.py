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


def infinite_plane(s:np.ndarray,r:np.ndarray, ratio=30.):
    """Create an "infinite" xy plane relative to eval point position.

    Parameters
    ----------
    s: np.array(3,)
        source position in 3D space

    r: np.array(3,)
        receiver position in 3D space

    ratio: float
        ratio between plane dimensions and
        distance between receiver and plane.

    Returns
    -------
    plane: sparapy.geometry.Polygon
        plane surface in format compatible with sparapy.

    """
    length = ratio*max(s[2],r[2]) # dimensions of plane's side

    centerx = s[0]-(s[0]-r[0])/2
    centery = s[1]-(s[1]-r[1])/2

    w = (abs(s[0]-r[0])+length)/2
    h = (abs(s[1]-r[1])+length)/2

    return  [
            sp.geometry.Polygon([ [centerx+w, centery+h, 0],
                                  [centerx-w, centery+h, 0],
                                  [centerx-w, centery-h, 0],
                                  [centerx+w, centery-h, 0]
                                ],
                                up_vector=np.array([1.,0.,0.]),
                                normal=np.array([0.,0.,1.]))
            ]

def get_histogram(source_pos=np.array([1.,1.,1.]),
                  receiver_pos=np.array([-1.,-1.,1.]),
                  sampling_rate=1000.,
                  h2ps_ratio=10.,
                  freq_bins=np.array([1000.]),
                  surfs=None
                  ):
    """Generate histogram of infinite plane scenario.

    Parameters
    ----------
    receiver_pos: np.ndarray((3,), dtype=float)
        receiver position in space.
        *z>0!*

    source_pos: np.ndarray((3,), dtype=float)
        source position in space.
        *z>0!*

    h2ps_ratio: float
        ratio of patch size relative to maximum
        source/receiver height.

    """
    ## BASIC STUFF ##
    #generate "infinite" plane
    if surfs is None:
        surfs = infinite_plane(r=receiver_pos,s=source_pos)

    #determine patch size based on input ratio
    patch_size = h2ps_ratio*max(receiver_pos[2],source_pos[2])

    #simulation parameters
    speed_of_sound = 346.18
    max_sound_path_length = np.sqrt(
        (surfs[0].pts[0]-surfs[0].pts[2])[0]**2 +
        (surfs[0].pts[0]-surfs[0].pts[2])[1]**2 +
        max(receiver_pos[2],source_pos[2])**2
    )
    max_histogram_length = .75*max_sound_path_length/speed_of_sound
    
    ## PREPARE RADIOSITY SIMULATION ##
    #initialize radiosity class instance based on plane "radi"
    radi = sp.DRadiosityFast.from_polygon(surfs,patch_size)

    #set scattering distribution (lambertian surface)
    scattering_data = pf.FrequencyData(
                np.ones((1, 1, freq_bins.size)), freq_bins)
    radi.set_wall_scattering(np.arange(1), scattering_data,
                             sources=pf.Coordinates(source_pos[0],source_pos[1],source_pos[2]),
                             receivers=pf.Coordinates(receiver_pos[0],receiver_pos[1],receiver_pos[2]))

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
        max_depth=1
    )
    
    #sum energy from all the patches at each time stamp.
    histogram = np.sum(radi.collect_receiver_energy(
                                    receiver_pos=receiver_pos,
                                    speed_of_sound=speed_of_sound,
                                    histogram_time_resolution=1/sampling_rate
                                    ),axis=1)[0]
    return histogram