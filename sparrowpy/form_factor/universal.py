"""methods for universal form factor calculation."""
import numpy as np
import sparrowpy.geometry as geom
from sparrowpy.form_factor import integration
try:
    import numba
    prange = numba.prange
except ImportError:
    numba = None
    prange = range

def patch2patch_ff_universal(patches_points: np.ndarray,
                             patches_normals: np.ndarray,
                             patches_areas: np.ndarray,
                             visible_patches:np.ndarray):
    """Calculate the form factors between patches (universal method).

    This method computes form factors between polygonal patches via numerical
    integration. Support is only guaranteed for convex polygons.
    Because this function relies on numpy, all patches must have
    the same number of sides.

    Parameters
    ----------
    patches_points : np.ndarray
        vertices of all patches of shape (n_patches, n_vertices, 3)
    patches_normals : np.ndarray
        normal vectors of all patches of shape (n_patches, 3)
    patches_areas : np.ndarray
        areas of all patches of shape (n_patches)
    visible_patches : np.ndarray
        index list of all visible patches combinations (n_combinations, 2)

    Returns
    -------
    form_factors : np.ndarray
        form factors between all patches of shape (n_patches, n_patches)
        note that just i_source < i_receiver are calculated since
        patches_areas[i] * ff[i, j] = patches_areas[j] * ff[j, i]

    """
    n_patches = patches_areas.shape[0]
    form_factors = np.zeros((n_patches, n_patches))

    for visID in prange(visible_patches.shape[0]):
        i = int(visible_patches[visID, 0])
        j = int(visible_patches[visID, 1])
        form_factors[i,j] = universal_form_factor(
                    patches_points[i], patches_normals[i], patches_areas[i],
                    patches_points[j], patches_normals[j])

    return form_factors

def universal_form_factor(source_pts: np.ndarray, source_normal: np.ndarray,
                     source_area: np.ndarray, receiver_pts: np.ndarray,
                     receiver_normal: np.ndarray,
                     ) -> float:
    """Return the form factor based on input patches geometry.

    Parameters
    ----------
    receiver_pts: np.ndarray
        receiver patch vertex coordinates (n_vertices,3)

    receiver_normal: np.ndarray
        receiver patch normal (3,)

    receiver_area: float
        receiver patch area

    source_pts: np.ndarray
        source patch vertex coordinates (n_vertices,3)

    source_normal: np.ndarray
        source patch normal (3,)

    source_area: float
        source patch area

    Returns
    -------
    form_factor: float
        form factor

    """
    if geom._coincidence_check(receiver_pts, source_pts):
        form_factor = integration.nusselt_integration(
                    patch_i=source_pts, patch_i_normal=source_normal,
                    patch_j=receiver_pts, patch_j_normal=receiver_normal,
                    nsamples=64)
    else:
        form_factor = integration.stokes_integration(patch_i=source_pts,
                                             patch_j=receiver_pts,
                                             patch_i_area=source_area)

    return form_factor

def _source2patch_energy_universal(
        source_position: np.ndarray, patches_center: np.ndarray,
        patches_points: np.ndarray, source_visibility: np.ndarray,
        air_attenuation:np.ndarray, n_bins:float):
    """Calculate the initial energy from the source.

    Parameters
    ----------
    source_position : np.ndarray
        source position of shape (3,)
    patches_center : np.ndarray
        center of all patches of shape (n_patches, 3)
    patches_points : np.ndarray
        vertices of all patches of shape (n_patches, n_points, 3)
    source_visibility : np.ndarray
        visibility condition between source and patches (n_patches)
    air_attenuation : np.ndarray
        air attenuation factor in Np/m (n_bins,)
    n_bins : float
        number of frequency bins.

    Returns
    -------
    energy : np.ndarray
        energy of all patches of shape (n_patches)
    distance : np.ndarray
        corresponding distance of all patches of shape (n_patches)

    """
    n_patches = patches_center.shape[0]
    energy = np.zeros((n_patches, n_bins))
    distance_out = np.zeros((n_patches, ))
    for j in prange(n_patches):
        if source_visibility[j]:
            source_pos = source_position.copy()
            receiver_pos = patches_center[j, :].copy()
            receiver_pts = patches_points[j, :, :].copy()

            distance_out[j] = np.linalg.norm(source_pos-receiver_pos)

            if air_attenuation is not None:
                energy[j,:] = np.exp(
                    -air_attenuation*distance_out[j])*integration.pt_solution(
                        point=source_pos, patch_points=receiver_pts,
                        mode="source")
            else:
                energy[j,:] = integration.pt_solution(
                    point=source_pos, patch_points=receiver_pts, mode="source")

    return (energy, distance_out)

def _patch2receiver_energy_universal(
        receiver_pos, patches_points, receiver_visibility):

    receiver_factor = np.zeros((patches_points.shape[0]))


    for i in prange(patches_points.shape[0]):
        if receiver_visibility[i]:
            receiver_factor[i] = integration.pt_solution(point=receiver_pos,
                            patch_points=patches_points[i,:], mode="receiver")

    return receiver_factor


if numba is not None:
    patch2patch_ff_universal = numba.njit(parallel=True)(
        patch2patch_ff_universal)
    universal_form_factor = numba.njit()(universal_form_factor)
    _source2patch_energy_universal = numba.njit(parallel=True)(
        _source2patch_energy_universal)
    _patch2receiver_energy_universal = numba.njit(parallel=True)(
        _patch2receiver_energy_universal)
