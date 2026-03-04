"""Methods and helpers for the form factor integration."""

try:
    import numba

    prange = numba.prange
except ImportError:
    numba = None
    prange = range
import numpy as np

import sparrowpy.geometry as geom


def load_stokes_integrand(
    i_bpoints: np.ndarray,
    j_bpoints: np.ndarray,
) -> np.ndarray:
    """Load all the stokes form function values between two patches.

    Parameters
    ----------
    i_bpoints: np.ndarray
        list of points in patch i boundary (n_boundary_points_i , 3)

    j_bpoints: np.ndarray
        list of points in patch j boundary (n_boundary_points_j , 3)

    Returns
    -------
    form_mat: np.ndarray
        f function value matrix (n_boundary_points_i , n_boundary_points_j)

    """
    eps = 1e-20
    form_mat = eps * np.ones((len(i_bpoints), len(j_bpoints)))

    for i in prange(i_bpoints.shape[0]):
        for j in prange(j_bpoints.shape[0]):
            rs = (np.linalg.norm(i_bpoints[i] - j_bpoints[j])) ** 2
            if rs > eps:
                form_mat[i][j] = rs

    return form_mat


def stokes_integration(
    patch_i: np.ndarray,
    patch_j: np.ndarray,
    patch_i_area: float,
) -> float:
    """Calculate an estimation of the form factor between two patches.

    Computationally integrates a modified form function over
    the boundaries of both patches.
    The modified form function follows Stokes' theorem.

    The modified form function integral is calculated using a
    polynomial approximation based on sampled values.

    Parameters
    ----------
    patch_i : np.ndarray
        vertex coordinates of patch i (n_vertices, 3)

    patch_i_area: float
       area of patch i

    patch_j : np.ndarray
        vertex coordinates of patch j (n_vertices, 3)

    source_area: float
        area of the source patch

    approx_order: int
        polynomial order of the form function integration estimation

    Returns
    -------
    float
    form factor between two patches

    """
    i_bpoints, i_conn, di = _sample_boundary_GL2(patch_i)
    j_bpoints, j_conn = _sample_boundary_regular(patch_j)

    subseci = np.zeros((i_conn.shape[1]))
    subsecj = np.zeros((j_conn.shape[1]))
    form_mat = np.zeros((i_bpoints.shape[0], j_bpoints.shape[0]))

    # first compute and store form function sample values
    form_mat = load_stokes_integrand(i_bpoints, j_bpoints)

    # double polynomial integration (per dimension (x,y,z))
    outer_integral = 0
    inner_integral = np.zeros((len(i_bpoints), len(j_bpoints[0])))

    for dim in prange(j_bpoints.shape[1]):  # for each dimension
        # integrate stokes integrand over each point on patch i boundary
        for i in range(
            i_bpoints.shape[0],
        ):  # for each point in patch i boundary
            for segj in j_conn:  # for each segment segj in patch j boundary
                xj = j_bpoints[segj][:, dim]

                if np.abs(xj[-1] - xj[0]) > 1e-6:
                    for k in range(len(segj)):
                        subsecj[k] = form_mat[i][segj[k]]

                    # analytical integration of the approx polynomials
                    inner_integral[i][dim] += _first_integration_analytical(
                        x=xj,
                        rsquared=subsecj,
                    )

        # integrate previously computed integral over patch i
        for segi in i_conn:  # for each segment segi in patch i boundary
            xi = i_bpoints[segi][:, dim]

            if np.abs(xi[-1] - xi[0]) > 1e-6:
                for k in range(len(segi)):
                    subseci[k] = inner_integral[segi[k]][dim]

                # gauss-legendre integration (2nd order)
                outer_integral += di * np.dot(
                    subseci,
                    np.array(
                        [
                            0.8888888888888888,
                            0.5555555555555556,
                            0.5555555555555556,
                        ],
                    ),
                )

    return np.abs(outer_integral / (2 * np.pi * patch_i_area))


# ////////////////////////////////////////////////////////////////////////////#
###############################################################################
### point-to-patch and patch-to-point
def pt_solution(point: np.ndarray, patch_points: np.ndarray, mode="source"):
    """Calculate the geometric factor between a point and a patch.

    applies a modified version of the Nusselt analogue,
    transformed for a -point- source rather than differential surface element.

    Parameters
    ----------
    point: np.ndarray
        source or receiver point

    patch_points: np.ndarray
        vertex coordinates of the patch

    mode: string
        determines if point is acting as a source ('source')
        or as a receiver ('receiver')

    Returns
    -------
    geometric factor

    """
    if mode == "receiver":
        source_area = geom._polygon_area(patch_points)
    elif mode == "source":
        source_area = 4

    npoints = patch_points.shape[0]

    interior_angle_sum = 0

    patch_onsphere = np.zeros_like(patch_points)

    for i in range(npoints):
        patch_onsphere[i] = (patch_points[i] - point) / np.linalg.norm(
            patch_points[i] - point,
        )

    for i in range(npoints):
        v0 = geom._sphere_tangent_vector(
            patch_onsphere[i],
            patch_onsphere[(i - 1) % npoints],
        )
        v1 = geom._sphere_tangent_vector(
            patch_onsphere[i],
            patch_onsphere[(i + 1) % npoints],
        )

        interior_angle_sum += np.arccos(np.dot(v0, v1))

    factor = interior_angle_sum - (len(patch_points) - 2) * np.pi

    return factor / (np.pi * source_area)


###################################################
# integration
################# 1D , polynomial
def _poly_estimation_Lagrange(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate Lagrange polynomial coefficients based on sample points.

    Computes coefficients of a polynomial curve passing through points (x,y)
    the order of the polynomial depends on the number of sample points
    input in the function. Uses the Lagrange method to estimate the polynomial.
        ex. a polynomial P estimated with 4 sample points:
            P4(x) = b[0]*x**3 + b[1]*x**2 + b[2]*x + b[3] = y

    Parameters
    ----------
    x: np.ndarray
        sample x-values
    y: np.ndarray
        sample y-values

    Returns
    -------
    b: np.ndarray
        polynomial coefficients

    """
    xmat = np.empty((len(x), len(x)))

    if np.abs(x[-1] - x[0]) < 1e-6:
        b = np.zeros(len(x))
    else:
        for i, xi in enumerate(x):
            for o in range(len(x)):
                xmat[i, len(x) - 1 - o] = xi**o

        b = geom._matrix_vector_product(np.linalg.inv(xmat), y)

    return b


def _first_integration_analytical(x: np.ndarray, rsquared: np.ndarray):
    """Calculate first integral analytically."""

    a = np.zeros((2,))
    g = np.zeros((2,))

    a = np.log(np.sqrt(rsquared)) * x

    poly_factors = _poly_estimation_Lagrange(x=x, y=rsquared)

    poly_factors[poly_factors == 0] = 1e-20

    g = _g_integral(abc=poly_factors, x=x)

    integral = a + g

    return integral[-1] - integral[0]


def _g_integral(abc: np.ndarray, x: np.ndarray):
    """Calculate second half of integral."""

    g = np.empty_like(x)
    a = abc[0]
    b = abc[1]
    c = abc[2]

    k = 4 * a * c - b**2
    kk = a * x**2 + b * x + c
    if k <= 0:
        k = 1e-20

    kk[kk <= 0] = 1e-20

    aa = 2 * np.sqrt(k)
    bb = np.arctan((2 * a * x + b) / np.sqrt(k))
    cc = b * np.log(kk) - 4 * a * x

    g = (aa * bb + cc) / 4 * a

    return g


####################################################
# sampling
################# boundary
def _sample_boundary_GL2(el: np.ndarray):
    """Sample element boundary after order 2 gauss-legendre quadrature."""
    conn = np.empty((el.shape[0], 3), dtype=numba.int16)
    pts = np.empty((el.shape[0] * (3), el.shape[1]))
    x = np.array([0, 1.3416407864998738, -1.3416407864998738])

    for i in prange(el.shape[0]):
        u = (el[(i + 1) % el.shape[0]] - el[i]) / 2
        v = (el[(i + 1) % el.shape[0]] + el[i]) / 2

        pts[i * 3 : (i + 1) * 3, :] = np.outer(x, u) + v

        conn[i] = i * x.shape[0] + np.arange(x.shape[0])

    return pts, conn.astype(np.int16), u


def _sample_boundary_regular(el: np.ndarray):
    """Sample points on the boundary of a patch at fractional intervals.

    returns an array of points on the patch boundary (pts)
                                        and a connectivity array (conn)
    which stores a list of ordered indices of the points
    found on the same boundary segment.

    Parameters
    ----------
    el : geometry.Polygon object
        patch to sample

    npoints : int
        number of sample points per boundary segment (minimum 2)

    Returns
    -------
    pts: np.ndarray
        boundary sample points

    conn: np.ndarray(int)
        indices of pts corresponding to boundary segments
        (each row corresponds to the points in a single segment)

    """
    n_div = 2

    pts = np.empty((el.shape[0] * 2, el.shape[1]))
    conn = np.empty((el.shape[0], 3), dtype=np.int16)

    for i in range(el.shape[0]):
        conn[i][0] = (i * n_div) % (n_div * el.shape[0])
        conn[i][-1] = (i * n_div + n_div) % (n_div * el.shape[0])

        for ii in range(0, n_div):
            pts[i * n_div + ii, :] = (
                el[i] + ii * (el[(i + 1) % el.shape[0]] - el[i]) / n_div
            )

            conn[i][ii] = (i * n_div + ii) % (n_div * el.shape[0])

    return pts, conn.astype(np.int16)


if numba is not None:
    pt_solution = numba.njit(parallel=True)(pt_solution)
    stokes_integration = numba.njit(parallel=False)(stokes_integration)
    load_stokes_integrand = numba.njit(parallel=True)(load_stokes_integrand)
    _first_integration_analytical = numba.njit()(_first_integration_analytical)
    _g_integral = numba.njit()(_g_integral)
    _sample_boundary_regular = numba.njit()(_sample_boundary_regular)
    _poly_estimation_Lagrange = numba.njit(_sample_boundary_regular)
    _sample_boundary_GL2 = numba.njit()(_sample_boundary_GL2)
