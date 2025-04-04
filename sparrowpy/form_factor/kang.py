"""Form factor calculation for radiosity."""
try:
    import numba
    prange = numba.prange
except ImportError:
    numba = None
    prange = range
import numpy as np

## form factor matrix assembly

def patch2patch_ff_kang(
        patches_center:np.ndarray, patches_normal:np.ndarray,
        patches_size:np.ndarray, visible_patches:np.ndarray) -> np.ndarray:
    """Calculate the form factors between patches.

    Parameters
    ----------
    patches_center : np.ndarray
        center points of all patches of shape (n_patches, 3)
    patches_normal : np.ndarray
        normal vectors of all patches of shape (n_patches, 3)
    patches_size : np.ndarray
        size of all patches of shape (n_patches, 3)
    visible_patches : np.ndarray
        index list of all visible patches combinations (n_combinations, 2)

    Returns
    -------
    form_factors : np.ndarray
        form factors between all patches of shape (n_patches, n_patches)
        note that just i_source < i_receiver are calculated ff[i, j] = ff[j, i]

    """
    n_patches = patches_center.shape[0]
    form_factors = np.zeros((n_patches, n_patches))
    for i in prange(visible_patches.shape[0]):
        i_source = int(visible_patches[i, 0])
        i_receiver = int(visible_patches[i, 1])
        source_center = patches_center[i_source]
        source_normal = patches_normal[i_source]
        receiver_center = patches_center[i_receiver]
        # calculation of form factors
        receiver_normal = patches_normal[i_receiver]
        dot_product = np.dot(receiver_normal, source_normal)

        if dot_product == 0:  # orthogonal

            if np.abs(source_normal[0]) > 1e-5:
                idx_source = {2, 1}
                dl = source_center[2]
                dm = source_center[1]
                dd_l = patches_size[i_source, 2]
                dd_m = patches_size[i_source, 1]
            elif np.abs(source_normal[1]) > 1e-5:
                idx_source = {2, 0}
                dl = source_center[2]
                dm = source_center[0]
                dd_l = patches_size[i_source, 2]
                dd_m = patches_size[i_source, 0]
            elif np.abs(source_normal[2]) > 1e-5:
                idx_source = {0, 1}
                dl = source_center[1]
                dm = source_center[0]
                dd_l = patches_size[i_source, 1]
                dd_m = patches_size[i_source, 0]

            if np.abs(receiver_normal[0]) > 1e-5:
                idx_l = 1 if 1 in idx_source else 2
                idx_s = 0
                idx_r = 2 if 1 in idx_source else 1
                dl_prime = receiver_center[1]
                dn_prime = receiver_center[2]
            elif np.abs(receiver_normal[1]) > 1e-5:
                idx_l = 0 if 0 in idx_source else 2
                idx_s = 1
                idx_r = 2 if 0 in idx_source else 0
                dl_prime = receiver_center[0]
                dn_prime = receiver_center[2]
            elif np.abs(receiver_normal[2]) > 1e-5:
                idx_l = 0 if 0 in idx_source else 1
                idx_s = 2
                idx_r = 1 if 0 in idx_source else 0
                dl_prime = receiver_center[1]
                dn_prime = receiver_center[0]

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
            if np.abs(receiver_normal[0]) > 1e-5:
                dl = receiver_center[1]
                dm = receiver_center[0]
                dn = receiver_center[2]
                dl_prime = source_center[1]
                dm_prime = source_center[0]
                dn_prime = source_center[2]
                dd_l = patches_size[i_source, 1]
                if patches_size.shape[1] > 2:
                    dd_n = patches_size[i_source, 2]
                else:
                    dd_n = patches_size[i_source, 1]
            elif np.abs(receiver_normal[1]) > 1e-5:
                dl = receiver_center[0]
                dm = receiver_center[1]
                dn = receiver_center[2]
                dl_prime = source_center[0]
                dm_prime = source_center[1]
                dn_prime = source_center[2]
                dd_l = patches_size[i_source, 0]
                if patches_size.shape[1] > 2:
                    dd_n = patches_size[i_source, 2]
                else:
                    dd_n = patches_size[i_source, 1]
            elif np.abs(receiver_normal[2]) > 1e-5:
                dl = receiver_center[1]
                dm = receiver_center[2]
                dn = receiver_center[0]
                dl_prime = source_center[1]
                dm_prime = source_center[2]
                dn_prime = source_center[0]
                dd_l = patches_size[i_source, 1]
                dd_n = patches_size[i_source, 0]

            d = np.sqrt(
                ( (dl - dl_prime) ** 2 ) +
                ( (dn - dn_prime) ** 2 ) +
                ( (dm - dm_prime) ** 2 ) )
            # Equation 16
            ff =  ( dd_l * dd_n * ( (
                dm-dm_prime) ** 2 ) ) / ( np.pi * ( d**4 ) )

        form_factors[i_source, i_receiver] = ff
    return form_factors

def patch2receiver_ff_kang(
        patch_receiver_distance, patches_normal):
    receiver_factor = np.empty((
        patch_receiver_distance.shape[0],))
    for i in range(patch_receiver_distance.shape[0]):
        R = np.sqrt(np.sum((patch_receiver_distance[i, :]**2)))

        cos_xi = np.abs(np.sum(
            patches_normal[i, :]*np.abs(patch_receiver_distance[i, :]))) / R

        # Equation 20
        receiver_factor[i] = cos_xi / (np.pi * R**2)

    return receiver_factor

if numba is not None:
    patch2patch_ff_kang = numba.njit(parallel=True)(patch2patch_ff_kang)
    patch2receiver_ff_kang = numba.njit()(patch2receiver_ff_kang)
