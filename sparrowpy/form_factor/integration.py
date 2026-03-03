"""Methods and helpers for the form factor integration."""

try:
    import numba

    prange = numba.prange
except ImportError:
    numba = None
    prange = range
import numpy as np

import sparrowpy.geometry as geom


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

    npoints = len(patch_points)

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


def _poly_integration(c: np.ndarray, x: np.ndarray) -> float:
    """Integrate a polynomial curve.

    polynomial defined defined between x[0] and x[-1]
    with coefficients c
        ex. for a quadratic curve P2:
            P2(x) = c[0]*x**2 + c[1]*x + c[2]

    Parameters
    ----------
    c: np.ndarray
        polynomial coefficients
    x: np.ndarray
        sample points

    Returns
    -------
    out: float
        polynomial integral

    """
    out = 0

    for i in range(len(c)):
        out += c[i] * x[-1] ** (len(c) - i) / (len(c) - i)
        out -= c[i] * x[0] ** (len(c) - i) / (len(c) - i)

    return out


def _lagrange_integral(x: np.ndarray, y: np.ndarray, d):
    """Integrate samples by Lagrange polynomial estimation.

    Input function is approximated by a Lagrange polynomial
    and integrated. The order of the polynomial approximation
    is defined by the number of samples (order=n_samples-1)
    Approximations up to polynomial order 6 employ closed
    Newton-Cotes formulas. If the samples are not equally spaced,
    the generalized approach is used.

    Parameters
    ----------
    x: np.ndarray (n_samples,)
        sample x-coordinates.
    y: np.ndarray (n_samples,)
        sample y-coordinates.

    Returns
    -------
    out: integral of the approximated function.
    """
    if x.shape[0] != y.shape[0]:
        ValueError("x and y arrays must have the same length!")
    if x.shape[1:] != (1,):
        ValueError(f"x array shape {x.shape} must be one-dimensional")
    if y.shape[1:] != (1,):
        ValueError(f"y array shape {y.shape} must be one-dimensional")
    if x.shape[0] == 1:
        ValueError("Impossible to evaluate integral with a single sample.")

    o = y.shape[0] - 1
    steps = x[1:] - x[:-1]

    if o < 7 and (steps == steps[0]).all():
        match o:
            case 1:
                # Trapezoidal rule
                NC_coefs = 0.5 * np.array([1.0, 1.0])
            case 2:
                # Simpson's rule
                NC_coefs = 1 / 3 * np.array([1.0, 4.0, 1.0])
            case 3:
                # Simpson's 3/8 rule
                NC_coefs = 3 / 8 * np.array([1.0, 3.0, 3.0, 1.0])
            case 4:
                # Boole's rule
                NC_coefs = 2 / 45 * np.array([7.0, 32.0, 12.0, 32.0, 7.0])
            case 5:
                NC_coefs = (
                    5 / 288 * np.array([19.0, 75.0, 50.0, 50.0, 75.0, 19.0])
                )
            case 6:
                NC_coefs = (
                    1
                    / 140
                    * np.array([41.0, 216.0, 27.0, 272.0, 27.0, 216.0, 41])
                )

        out = d / o * np.dot(y, NC_coefs)
    else:
        # generalized implementation
        poly_coefs = _poly_estimation_Lagrange(x, y)
        out = _poly_integration(poly_coefs, x)

    return out


def _gauss_legendre_integral(x: np.ndarray, y: np.ndarray, d):
    """Integrate samples by Legendre polynomial estimation.

    Input function is approximated by a Lagrange polynomial
    and integrated. The order of the polynomial approximation
    is defined by the number of samples (order=n_samples-1)
    Approximations up to polynomial order 6 employ closed
    Newton-Cotes formulas. If the samples are not equally spaced,
    the generalized approach is used.

    Parameters
    ----------
    x: np.ndarray (n_samples,)
        sample x-coordinates.
    y: np.ndarray (n_samples,)
        sample y-coordinates.

    Returns
    -------
    out: integral of the approximated function.
    """
    o = y.shape[0] - 1

    match o:
        case 1:
            GL_coefs = np.array([1.0, 1.0])
        case 2:
            GL_coefs = np.array([8 / 9, 5 / 9, 5 / 9])
        case 3:
            a = (18 + 30**0.5) / 36
            b = (18 - 30**0.5) / 36
            GL_coefs = 3 / 8 * np.array([a, a, b, b])
        case 4:
            a = 128 / 225
            b = (322 + 13 * 70**0.5) / 900
            c = (322 - 13 * 70**0.5) / 900
            GL_coefs = 2 / 45 * np.array([a, b, b, c, c])

    out = d / 2 * np.dot(y, np.abs(GL_coefs))

    return out


def _first_integration_analytical(
    x: np.ndarray,
    rsquared: np.ndarray,
    delta: np.ndarray,
):
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

    A = 2 * np.sqrt(k)
    B = np.arctan((2 * a * x + b) / np.sqrt(k))
    C = b * np.log(kk) - 4 * a * x

    g = (A * B + C) / 4 * a

    return g


# /////////////////////////////////////////////////////////////////////////////////////#
#######################################################################################


def _load_stokes_integrand(
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
            form_mat[i][j] = np.log(
                np.linalg.norm(i_bpoints[i] - j_bpoints[j]),
            )

    return form_mat


def _load_analytical_integrand(
    i_bpoints: np.ndarray,
    j_bpoints: np.ndarray,
) -> np.ndarray:
    """Load all the analytical form function values between two patches.

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


def _load_Lambert_integrand(
    i_points: np.ndarray,
    i_normal: np.ndarray,
    j_points: np.ndarray,
    j_normal: np.ndarray,
) -> np.ndarray:
    """Load all the Lambert cosine law values between two patches.

    Parameters
    ----------
    i_points: np.ndarray
        list of points in patch i surface (n_points_i , 3)

    j_points: np.ndarray
        list of points in patch j surface (n_points_j , 3)

    Returns
    -------
    form_mat: np.ndarray
        f function value matrix (n_boundary_points_i , n_boundary_points_j)

    """
    form_mat = np.zeros((i_points.shape[0], j_points.shape[0]))
    for i, ip in enumerate(i_points):
        for j, jp in enumerate(j_points):
            r = ip - jp
            if np.linalg.norm(r) == 0:
                form_mat[i, j] = 0
            else:
                rr = np.linalg.norm(r)

                form_mat[i, j] = (
                    np.dot(-r, i_normal) * np.dot(r, j_normal) / rr**4
                )

    return form_mat


def contour_ff(
    patch_i: np.ndarray,
    patch_j: np.ndarray,
    patch_i_area: float,
    inner_style="analytical",
    outer_style="poly_NC",
    order=2,
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

    integrand_fcn: function handle
        function handle to integrand function

    Returns
    -------
    float
    form factor between patch i and j

    """
    if inner_style == "analytical":
        j_bpoints, j_conn = _sample_boundary(patch_j, npoints=3)
        integrand_fcn = _load_analytical_integrand

    elif inner_style == "poly_NC":
        j_bpoints, j_conn = _sample_boundary(
            patch_j,
            npoints=order + 1,
            style="regular",
        )
        integrand_fcn = _load_stokes_integrand

    elif inner_style == "poly_GL":
        j_bpoints, j_conn = _sample_boundary(
            patch_j,
            npoints=order + 1,
            style="GL",
        )
        integrand_fcn = _load_stokes_integrand

    else:
        ValueError(f"{inner_style} integration method unknown")

    if outer_style == "poly_NC":
        i_bpoints, i_conn = _sample_boundary(
            patch_i,
            npoints=order + 1,
            style="regular",
        )
    elif outer_style == "poly_GL":
        i_bpoints, i_conn = _sample_boundary(
            patch_i,
            npoints=order + 1,
            style="GL",
        )

    integrand = np.zeros((i_bpoints.shape[0], j_bpoints.shape[0]))

    # first compute and store form function sample values
    integrand = integrand_fcn(i_bpoints, j_bpoints)

    # double polynomial integration (per dimension (x,y,z))
    outer_integral = 0
    inner_integral = np.zeros((1, i_bpoints.shape[0]))

    for dim in range(len(j_bpoints[0])):  # for each dimension x,y,x
        # integrate over target patch
        inner_integral = contour_integration(
            patch_coords=j_bpoints[:, dim],
            patch_conn=j_conn,
            integrand=integrand,
            delta=patch_j[:, dim],
            style=inner_style,
        )

        # second integral over source patch
        outer_integral += contour_integration(
            patch_coords=i_bpoints[:, dim],
            patch_conn=i_conn,
            integrand=inner_integral,
            delta=patch_i[:, dim],
            style=outer_style,
        )

    return np.abs(outer_integral[0, 0] / (2 * np.pi * patch_i_area))


def contour_integration(
    patch_coords: np.ndarray,
    patch_conn: np.ndarray,
    integrand: np.ndarray,
    delta: np.ndarray,
    style="analytical",
):
    """Integrate a sampled integrand over a patch contour.

    Parameters
    ----------
    patch_coords : np.ndarray
        list of contour sample coordinates of patch (n_samples, 3)

    patch_conn: list
       sample connectivity for each of the contour segments (n_segments,...,)

    integrand : np.ndarray
        matrix of integrand samples (..., n_samples)

    int_function: function handle
        integration approximation function handle

    Returns
    -------
    float
    form factor between patch i and j

    """

    subsecj = np.zeros((patch_conn.shape[1]))

    out = np.zeros(((1,) + integrand.shape[:1]))

    d = np.roll(delta, -1) - delta
    if style == "analytical":
        for i in range(integrand.shape[0]):  # for each eval point
            for ii, seg in enumerate(
                patch_conn,
            ):  # for each segment seg in patch  boundary
                x = patch_coords[seg][:]

                if np.abs(d[ii]) > 1e-3:
                    for k in range(seg.shape[0]):
                        subsecj[k] = integrand[i][seg[k]]

                        # add separate integral approx contributions
                        out[:, i] += _first_integration_analytical(
                            x,
                            subsecj,
                            d[ii],
                        )

    elif style == "poly_NC":
        for i in range(integrand.shape[0]):  # for each eval point
            for ii, seg in enumerate(
                patch_conn,
            ):  # for each segment seg in patch  boundary
                x = patch_coords[seg][:]

                if np.abs(d[ii]) > 1e-3:
                    for k in range(len(seg)):
                        subsecj[k] = integrand[i][seg[k]]

                        # add separate integral approx contributions
                        out[:, i] += _lagrange_integral(
                            x,
                            subsecj,
                            d[ii],
                        )
    elif style == "poly_GL":
        for i in range(integrand.shape[0]):  # for each eval point
            for ii, seg in enumerate(
                patch_conn,
            ):  # for each segment seg in patch  boundary
                x = patch_coords[seg][:]

                if np.abs(d[ii]) > 1e-3:
                    for k in range(len(seg)):
                        subsecj[k] = integrand[0][0]

                        # add separate integral approx contributions
                        out[:, i] += _gauss_legendre_integral(
                            x,
                            subsecj,
                            d[ii],
                        )

    return out


def surface_ff_Nusselt(
    patch_i: np.ndarray,
    patch_j: np.ndarray,
    patch_i_normal: np.ndarray,
    patch_j_normal: np.ndarray,
    patch_i_area: float,
    nsamples=9,
    style="reg_surf",
) -> float:
    """Estimate form factors based on double surface integration.

    Integrates the differential form factor (Nusselt analogue output)
    over the surface of the source patch

    Parameters
    ----------
    patch_i: np.ndarray
        vertex coordinates of the source patch

    patch_j: np.ndarray
        vertex coordinates of the receiver patch

    patch_i_normal: np.ndarray
        source patch normal (3,)

    patch_j_normal: np.ndarray
        receiver patch normal (3,)

    patch_i_area: float
        source patch area

    patch_j_area: float
        receiver patch area

    nsamples: int
        number of receiver surface samples for integration

    random: bool
        determines the distribution of the samples on patch_i surface
        if True, the samples are randomly distributed in a uniform way
        if False, a regular sampling of the surface is performed

    Returns
    -------
    out: float
        form factor between patches i and j

    """

    p0_array, nu, nv, stepx, stepy = _surf_sampling(
        patch_i,
        npoints=nsamples,
        style=style,
    )

    int_int = np.empty((nu, nv))

    for k in prange(p0_array.shape[0]):
        int_int[int(k / nu), k % nv] = nusselt_analog(
            surf_origin=p0_array[k],
            surf_normal=patch_i_normal,
            patch_points=patch_j,
            patch_normal=patch_j_normal,
        )

    out = _surface_integral(
        integrand=int_int,
        steps=(stepx, stepy),
        nn=(nu, nv),
        area=patch_i_area,
        style=style,
    )

    out *= 1 / (np.pi * patch_i_area)

    return out


def surface_ff_naive(
    patch_i: np.ndarray,
    patch_j: np.ndarray,
    patch_i_normal: np.ndarray,
    patch_j_normal: np.ndarray,
    patch_i_area: float,
    patch_j_area: float,
    nsamples=9,
    style="random",
) -> float:
    """Estimate form factors based on double surface integration.

    Integrates both surfaces by sampling points across the surfaces

    Parameters
    ----------
    patch_i: np.ndarray
        vertex coordinates of the source patch

    patch_j: np.ndarray
        vertex coordinates of the receiver patch

    patch_i_normal: np.ndarray
        source patch normal (3,)

    patch_j_normal: np.ndarray
        receiver patch normal (3,)

    patch_i_area: float
        source patch area

    patch_j_area: float
        receiver patch area

    nsamples: int
        number of receiver surface samples for integration

    random: bool
        determines the distribution of the samples on patch_i surface
        if True, the samples are randomly distributed in a uniform way
        if False, a regular sampling of the surface is performed

    Returns
    -------
    out: float
        form factor between patches i and j

    """

    pi_array, niu, niv, stepix, stepiy = _surf_sampling(
        patch_i,
        nsamples,
        style,
    )
    pj_array, nju, njv, stepjx, stepjy = _surf_sampling(
        patch_j,
        nsamples,
        style,
    )

    lambert = _load_Lambert_integrand(
        i_points=pi_array,
        i_normal=patch_i_normal,
        j_points=pj_array,
        j_normal=patch_j_normal,
    )

    int_int = np.empty(pi_array.shape[:-1])

    for i in prange(pi_array.shape[0]):
        if pi_array.ndim > 2:
            int_int[i] = _surface_integral(
                integrand=lambert[
                    int(i / pi_array.shape[0]) + i % pi_array.shape[1],
                    :,
                ],
                steps=(stepjx, stepjy),
                nn=(nju, njv),
                area=patch_j_area,
                style=style,
            )
        else:
            int_int[i] = _surface_integral(
                integrand=lambert[i, :],
                steps=(stepjx, stepjy),
                nn=(nju, njv),
                area=patch_j_area,
                style=style,
            )

    out = _surface_integral(
        integrand=int_int,
        steps=(stepix, stepiy),
        nn=(niu, niv),
        area=patch_i_area,
        style=style,
    )

    out *= 1 / (np.pi * patch_i_area)

    return out


def _surface_integral(
    integrand: np.ndarray,
    nn=(1, 1),
    steps=None,
    area=None,
    style="random",
):
    """Numerically integrate a given integrand over a surface.

    Parameters
    ----------
    integrand: np.ndarray (n_x_samples,n_y_samples) or (n_samples)
        integrand samples. The shape of the array determines if the
        integration employs 2-D Newton-Cotes formulae or a simple
        center-point average approach.

    steps: tuple (int,int)
        step sizes for the 2D Newton-Cotes formulae.

    Returns
    -------
    out: float
        integration result.
    """

    if style == "random" or style == "regular":
        out = np.sum(integrand) * area / integrand.shape[0]
    else:
        integrand = integrand.reshape((int(nn[0]), int(nn[1])))
        in1 = np.empty((integrand.shape[0]))

        x2 = np.arange(0, integrand.shape[1] * steps[1], steps[1])
        x1 = np.arange(0, integrand.shape[0] * steps[0], steps[0])

        if style == "poly_NC":
            for i in prange(integrand.shape[0]):
                in1[i] = _lagrange_integral(x1, integrand[i, :], d=steps[0])
            out = _lagrange_integral(x2, in1, d=steps[1])
        elif style == "poly_GL":
            for i in prange(integrand.shape[0]):
                in1[i] = _gauss_legendre_integral(
                    x1,
                    integrand[i, :],
                    d=steps[0],
                )
            out = _gauss_legendre_integral(x2, in1, d=steps[1])

    return out


def nusselt_analog(
    surf_origin,
    surf_normal,
    patch_points,
    patch_normal,
) -> float:
    """Calculate the Nusselt analog for a single point.

    Projects a given receiver patch onto a hemisphere centered around a point
    on a source patch surface.
    The hemispherical projection is then projected onto the source patch plane.
    The area of this projection relative to the unit circle area is the
    differential form factor between the two patches.

    Parameters
    ----------
    surf_origin : np.ndarray
        point on source patch for differential form factor evaluation (3,)
        (global origin)

    surf_normal : np.ndarray
        normal of source patch (3,)

    patch_points : np.ndarray
        vertex coordinates of the receiver patch (n_vertices, 3)

    patch_normal: np.ndarray
        normal of receiver patch (3,)

    Returns
    -------
    Nusselt analog factor
    (differential form factor)

    """
    boundary_points, connectivity = _sample_boundary(patch_points, npoints=3)

    hand = np.sign(
        np.dot(
            np.cross(
                patch_points[1] - patch_points[0],
                patch_points[2] - patch_points[1],
            ),
            patch_normal,
        ),
    )

    curved_area = 0

    sphPts = np.empty_like(boundary_points)
    projPts = np.empty_like(boundary_points)
    plnPts = np.empty(shape=(len(boundary_points), 2))

    for ii in prange(len(boundary_points)):
        # patch j points projected on the hemisphere
        if np.linalg.norm(boundary_points[ii] - surf_origin) != 0:
            sphPts[ii] = (boundary_points[ii] - surf_origin) / np.linalg.norm(
                boundary_points[ii] - surf_origin,
            )
        else:
            sphPts[ii] = 0

    rotmat = geom._rotation_matrix(n_in=surf_normal)

    for ii in prange(len(sphPts)):
        # points on the hemisphere projected onto patch plane
        plnPts[ii, :] = geom._matrix_vector_product(
            matrix=rotmat,
            vector=sphPts[ii],
        )[:-1]
        projPts[ii, :-1] = plnPts[ii, :]
        projPts[ii, -1] = 0.0

    big_poly = geom._polygon_area(projPts[0::2])

    segmt = np.empty_like(connectivity[0])

    leftseg = np.empty((3, 2))
    rightseg = np.empty((3, 2))

    for jj in prange(connectivity.shape[0]):
        segmt = connectivity[jj]

        if (
            np.linalg.norm(np.cross(projPts[segmt[-1]], projPts[segmt[0]]))
            > 1e-6
        ):
            # if the points on the segment span less than 90 degrees
            if np.dot(plnPts[segmt[-1]], plnPts[segmt[0]]) >= 1e-6:
                curved_area += _area_under_curve(plnPts[segmt], order=2)

            # if points span over 90º, additional sampling is required
            else:
                mpoint = (
                    sphPts[segmt[0]]
                    + (sphPts[segmt[-1]] - sphPts[segmt[0]]) / 2
                )

                # midpoint on the arc projected on the hemisphere
                marc = mpoint / np.linalg.norm(mpoint)
                a = sphPts[segmt[0]] + (marc - sphPts[segmt[0]]) / 2
                b = marc + (sphPts[segmt[-1]] - marc) / 2

                mpoint = geom._matrix_vector_product(
                    matrix=rotmat,
                    vector=mpoint,
                )[:-1]
                marc = geom._matrix_vector_product(matrix=rotmat, vector=marc)[
                    :-1
                ]
                a = a / np.linalg.norm(a)
                a = geom._matrix_vector_product(matrix=rotmat, vector=a)[:-1]

                b = b / np.linalg.norm(b)
                b = geom._matrix_vector_product(matrix=rotmat, vector=b)[:-1]

                linArea = (
                    np.linalg.norm(plnPts[segmt[-1]] - plnPts[segmt[0]])
                    * np.linalg.norm(mpoint - marc)
                    / 2
                )

                leftseg[0] = plnPts[segmt[0]]
                leftseg[1] = a
                leftseg[2] = marc

                rightseg[0] = marc
                rightseg[1] = b
                rightseg[2] = plnPts[segmt[-1]]

                left = _area_under_curve(leftseg, order=2)
                right = _area_under_curve(rightseg, order=2)
                curved_area += linArea * np.sign(left) + left + right

    return big_poly + hand * curved_area


################# surface areas


def _area_under_curve(ps: np.ndarray, order=2) -> float:
    """Calculate the area under a polynomial curve.

    Curve sampled by a finite number of points ps on a common plane.

    Parameters
    ----------
    ps : np.ndarray
        sample points

    order : int
        polynomial order of the curve

    Returns
    -------
    area: float
        area under curve

    """
    # the order of the curve may be overwritten depending on the sample size
    order = min(order, len(ps) - 1)

    # the vector between first and last sample (y==0) (new space's x axis)
    f = ps[-1] - ps[0]

    rotation_matrix = np.array([[f[0], f[1]], [-f[1], f[0]]]) / np.linalg.norm(
        f,
    )

    x = np.zeros(order + 1)
    y = np.zeros(order + 1)

    for k in range(1, order + 1):
        c = ps[k] - ps[0]  # translate point towards new origin

        # rotate point around origin to align with new axis
        cc = geom._matrix_vector_product(matrix=rotation_matrix, vector=c)
        x[k] = cc[0]
        y[k] = cc[1]

    area = _lagrange_integral(x, y, d=np.linalg.norm(f))

    return area


####################################################
# sampling
################# surface
def _surf_sampling(el: np.ndarray, npoints=10, style="random"):
    """Sample points on the surface of a patch using a regular distribution.

    over the directions defined by the patches' sides

    ! currently only supports triangular, rectangular, or parallelogram patches
    ! may not return exact number of requested points
                -- depends on the divisibility of the patch

    Parameters
    ----------
    el : geometry.Polygon object
        patch to sample

    npoints : int
        number of sample points to generate

    Returns
    -------
    out: np.ndarray
        list of sample points in patch el

    """
    u = el[1] - el[0]
    v = el[-1] - el[0]

    nu = int(np.sqrt(npoints))
    nv = nu

    match style:
        case "random":
            ptlist = np.empty((nu * nv, 3))
            for i in prange(nu * nv):
                s = np.random.uniform()
                t = np.random.uniform()
                ptlist[i] = s * u + t * v + el[0]

            stepx = None
            stepy = None

        case "regular":
            ptlist = np.empty((nu * nv, 3))
            stepv = 1 / (nv)
            stepu = 1 / (nu)

            tt = np.linspace(0.5 * stepv, 1 - 0.5 * stepv, nv)
            ts = np.linspace(0.5 * stepu, 1 - 0.5 * stepv, nu)
            ptlist = np.array([s * u + t * v + el[0] for t in tt for s in ts])

            stepx = None
            stepy = None

        case "poly_NC":
            nu = int(np.sqrt(npoints))
            nv = nu
            tt = np.linspace(0, 1, nv)
            ts = np.linspace(0, 1, nu)
            ptlist = np.array([s * u + t * v + el[0] for t in tt for s in ts])

            stepx = np.linalg.norm(u)
            stepy = np.linalg.norm(v)

        case "poly_GL":
            nu = int(np.sqrt(npoints))
            ptlist = np.empty((npoints, 3))
            xa, _ = _sample_boundary(el[0:2], npoints=nu, style="GL")
            xb, _ = _sample_boundary(
                np.roll(el[2:4], -1, axis=0),
                npoints=nu,
                style="GL",
            )

            for k in range(nu):
                ptlist[k * nu : (k + 1) * nu, :], _ = _sample_boundary(
                    el=np.array([xa[k], xb[k]]),
                    npoints=nu,
                    style="GL",
                )

            stepx = np.linalg.norm(el[1] - el[0])
            stepy = np.linalg.norm(el[2] - el[1])

    return ptlist, nu, nv, stepx, stepy


################# boundary
def _sample_boundary(el: np.ndarray, npoints=3, style="regular"):
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
    n_div = npoints - 1
    conn = np.empty((len(el), npoints), dtype=int)
    step = np.empty((len(el)))

    if style == "regular":
        pts = np.empty((len(el) * (npoints - 1), len(el[0])))
        for i in range(len(el)):
            conn[i][0] = (i * n_div) % (n_div * len(el))
            conn[i][-1] = (i * n_div + n_div) % (n_div * len(el))

            for ii in range(0, n_div):
                pts[i * n_div + ii, :] = (
                    el[i] + ii * (el[(i + 1) % len(el)] - el[i]) / n_div
                )

                conn[i][ii] = (i * n_div + ii) % (n_div * len(el))

    elif style == "GL":
        if len(el) > 2:
            pts = np.empty((len(el) * (npoints), len(el[0])))
            nsides = len(el)
        else:
            pts = np.empty((npoints, len(el[0])))
            nsides = 1

        x = _GL_samples(npoints)

        for i in range(nsides):
            u = (el[(i + 1) % len(el)] - el[i]) / 2
            v = (el[(i + 1) % len(el)] + el[i]) / 2

            pts[i * npoints : (i + 1) * npoints, :] = np.outer(x, u) + v

            conn[i] = i * len(x) + np.arange(len(x))

    return pts, conn.astype(int)


def _GL_samples(n: int):
    match n:
        case 2:
            a = 1 / 3**0.5
            x = np.array([-a, a])
        case 3:
            a = 0
            b = (3 / 5) ** 0.5
            x = np.array([a, b, -b])
        case 4:
            a = (3 / 7 - 2 / 7 * (6 / 5) ** 0.5) ** 0.5
            b = (3 / 7 + 2 / 7 * (6 / 5) ** 0.5) ** 0.5
            x = np.array([a, -a, b, -b])
        case 5:
            a = 0
            b = 1 / 3 * (5 - 2 * (10 / 7) ** 0.5) ** 0.5
            c = 1 / 3 * (5 + 2 * (10 / 7) ** 0.5) ** 0.5
            x = np.array([a, b, -b, c, -c])
        case _:
            raise ValueError(
                "No implementation of Gauss-Legendre"
                + "quadrature above 5 points.",
            )
    return x


if numba is not None:
    pt_solution = numba.njit(parallel=True)(pt_solution)
    contour_ff = numba.njit(parallel=False)(contour_ff)
    nusselt_analog = numba.njit(parallel=False)(nusselt_analog)
    _load_stokes_integrand = numba.njit(parallel=True)(_load_stokes_integrand)
    _load_analytical_integrand = numba.njit(parallel=True)(
        _load_analytical_integrand,
    )
    _poly_estimation_Lagrange = numba.njit()(_poly_estimation_Lagrange)
    _poly_integration = numba.njit()(_poly_integration)
    _first_integration_analytical = numba.njit()(_first_integration_analytical)
    _g_integral = numba.njit()(_g_integral)
    _sample_boundary = numba.njit()(_sample_boundary)
    _area_under_curve = numba.njit()(_area_under_curve)
    _lagrange_integral = numba.njit()(_lagrange_integral)
    _GL_samples = numba.njit()(_GL_samples)
    _gauss_legendre_integral = numba.njit()(_gauss_legendre_integral)
    contour_integration = numba.njit()(contour_integration)
