import time
import numpy as np
from scipy.integrate import dblquad
import sparrowpy.geometry as geom
from numpy.polynomial.legendre import leggauss

def pt_solution(point: np.ndarray, patch_points: np.ndarray, mode='source'):
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
    source_area = 4

    npoints = len(patch_points)

    interior_angle_sum = 0

    patch_onsphere = np.zeros_like(patch_points)

    for i in range(npoints):
        patch_onsphere[i]= ( (patch_points[i]-point) /
                              np.linalg.norm(patch_points[i]-point) )

    for i in range(npoints):

        v0 = geom._sphere_tangent_vector(patch_onsphere[i],
                                              patch_onsphere[(i-1)%npoints])
        v1 = geom._sphere_tangent_vector(patch_onsphere[i],
                                              patch_onsphere[(i+1)%npoints])

        interior_angle_sum += np.arccos(np.dot(v0,v1))

    factor = interior_angle_sum - (len(patch_points)-2)*np.pi 

    return factor / (np.pi*source_area)

#working scipy integrate function, but evidently 100x slower than current implementation
def point_patch_factor_dblquad(patch_points, point):
    edge1 = patch_points[1] - patch_points[0]
    edge2 = patch_points[3] - patch_points[0]
    patch_normal = np.cross(edge1, edge2)
    patch_normal = patch_normal / np.linalg.norm(patch_normal)

    # Ensure normal points toward receiver
    patch_center = np.mean(patch_points, axis=0)
    to_point = point - patch_center
    if np.dot(patch_normal, to_point) < 0:
        patch_normal = -patch_normal

    def integrand(u, v):
            # Bilinear interpolation to get point on patch
            patch_point = ((1-u)*(1-v)*patch_points[0] + 
                        u*(1-v)*patch_points[1] + 
                        u*v*patch_points[2] + 
                        (1-u)*v*patch_points[3])
            
            # Vector from patch to point
            r_vec = point - patch_point
            r_dist = np.linalg.norm(r_vec)
            
            if r_dist < 1e-10:  # avoid division by zero
                return 0.0
                
            r_dir = r_vec / r_dist
            
            # cos(θ_i): angle between patch normal and direction to point
            cos_theta_i = np.dot(patch_normal, r_dir)
            
            # Jacobian for area element
            dPdu = (-(1-v)*patch_points[0] + (1-v)*patch_points[1] + 
                    v*patch_points[2] - v*patch_points[3])
            dPdv = (-(1-u)*patch_points[0] - u*patch_points[1] + 
                    u*patch_points[2] + (1-u)*patch_points[3])
            
            dS = np.linalg.norm(np.cross(dPdu, dPdv))
            
            # Integrand: L * cos(θ_i) * cos(θ_o) * dS / (π * r²)
            return  1/np.pi * cos_theta_i * dS / r_dist**2


    # Integrate over unit square (u,v) ∈ [0,1] × [0,1]
    result = dblquad(integrand, 0, 1, 0, 1)
    return result

def point_patch_factor_leggaus(patch_points, point, N=8):
    edge1 = patch_points[1] - patch_points[0]
    edge2 = patch_points[3] - patch_points[0]
    patch_normal = np.cross(edge1, edge2)
    patch_normal = patch_normal / np.linalg.norm(patch_normal)

    # Ensure normal points toward receiver
    patch_center = np.mean(patch_points, axis=0)
    to_point = point - patch_center
    if np.dot(patch_normal, to_point) < 0:
        patch_normal = -patch_normal

    def integrand(u, v):
        # Bilinear interpolation to get point on patch
        patch_point = ((1-u)*(1-v)*patch_points[0] + 
                       u*(1-v)*patch_points[1] + 
                       u*v*patch_points[2] + 
                       (1-u)*v*patch_points[3])
        
        # Vector from patch to point
        r_vec = point - patch_point
        r_dist = np.linalg.norm(r_vec)
        
        if r_dist < 1e-10:  # avoid division by zero
            return 0.0
            
        r_dir = r_vec / r_dist
        
        # cos(θ_i): angle between patch normal and direction to point
        cos_theta_i = np.dot(patch_normal, r_dir)
        
        # Jacobian for area element
        dPdu = (-(1-v)*patch_points[0] + (1-v)*patch_points[1] + 
                v*patch_points[2] - v*patch_points[3])
        dPdv = (-(1-u)*patch_points[0] - u*patch_points[1] + 
                u*patch_points[2] + (1-u)*patch_points[3])
        
        dS = np.linalg.norm(np.cross(dPdu, dPdv))
        
        # Integrand: L * cos(θ_i) * cos(θ_o) * dS / (π * r²)
        return  1/np.pi * cos_theta_i * dS / r_dist**2

    # Legendre-Gauss quadrature nodes and weights
    u_nodes, u_weights = leggauss(N)
    v_nodes, v_weights = leggauss(N)
    # Map from [-1,1] to [0,1]:   t = 0.5*(x+1)
    total = 0.0
    for i in range(N):
        u = 0.5*(u_nodes[i]+1)
        wu = 0.5*u_weights[i]
        for j in range(N):
            v = 0.5*(v_nodes[j]+1)
            wv = 0.5*v_weights[j]
            total += integrand(u, v) * wu * wv
    return total

def point_patch_factor_leggaus_planar(patch_points, point, N=8):
    p0 = patch_points[0]
    edge_u = patch_points[1] - p0   # vector for u-direction
    edge_v = patch_points[3] - p0   # vector for v-direction

    # Constant area element (Jacobian)
    dS = np.linalg.norm(np.cross(edge_u, edge_v))
    patch_normal = np.cross(edge_u, edge_v)
    patch_normal /= np.linalg.norm(patch_normal)
    def integrand(u, v):
    # Affine mapping from (u,v) to 3D
        patch_point = p0 + u*edge_u + v*edge_v
        
        r_vec  = point - patch_point
        r_dist = np.linalg.norm(r_vec)
        if r_dist < 1e-10:
            return 0.0
        
        r_dir = r_vec / r_dist
        cos_theta = np.dot(patch_normal, r_dir)
        if cos_theta <= 0:  # back-facing
            return 0.0
        
        # constant dS, so no per-sample Jacobian
        return (1/np.pi) * cos_theta * dS / r_dist**2

    u_nodes, u_weights = leggauss(N)
    v_nodes, v_weights = leggauss(N)
    total = 0.0
    for i in range(N):
        u = 0.5*(u_nodes[i] + 1)
        wu = 0.5*u_weights[i]
        for j in range(N):
            v  = 0.5*(v_nodes[j] + 1)
            wv = 0.5*v_weights[j]
            total += integrand(u, v) * wu * wv
    return total

def point_patch_factor_mc_planar(patch_points, point, N=50, albedo=1.0):
    """
    Monte Carlo estimate of irradiance factor from a planar rectangular patch to a point,
    assuming a Lambertian BRDF with given albedo.

    patch_points : (4×3) array of corner coordinates [p0, p1, p2, p3]
    point        : (3,)   array of the receiver point
    N            : number of Monte Carlo samples
    albedo       : surface albedo (rho)

    Returns
    -------
    E_estimate : estimated E = ∫ f_r * cosθ * (dA / r^2)
    """
    # Setup patch geometry
    p0      = patch_points[0]
    edge_u  = patch_points[1] - p0
    edge_v  = patch_points[3] - p0
    # Compute patch normal and area
    patch_normal = np.cross(edge_u, edge_v)
    patch_area   = np.linalg.norm(patch_normal)
    patch_normal /= patch_area

    # Ensure normal faces the point
    patch_center = p0 + 0.5 * (edge_u + edge_v)
    if np.dot(patch_normal, point - patch_center) < 0:
        patch_normal = -patch_normal

    # Lambertian BRDF = rho/pi
    fr = albedo / np.pi

    total = 0.0
    for _ in range(N):
        # Uniform sample (u,v) ∈ [0,1]^2
        u, v = np.random.rand(), np.random.rand()
        # Map to 3D point on patch
        patch_pt = p0 + u*edge_u + v*edge_v

        # Vector from patch to point
        r_vec = point - patch_pt
        r2    = np.dot(r_vec, r_vec)
        if r2 < 1e-12:
            continue
        r_dir = r_vec / np.sqrt(r2)

        # Cosine term
        cos_theta = np.dot(patch_normal, r_dir)
        if cos_theta <= 0:
            continue

        # Contribution = f_r * cosθ * (dA / r²)
        total += fr * cos_theta * patch_area / r2

    # Average over samples
    return total / N


def point_patch_factor_dblquad_fix(patch_points: np.ndarray, point: np.ndarray, mode='source'):
    # Vectorized quadrature replacement for the slow dblquad implementation.
    # Uses Gauss-Legendre tensor product quadrature and NumPy broadcasting.
    N = 1  # quadrature order (adjust for speed/accuracy)
    eps = 1e-12

    edge1 = patch_points[1] - patch_points[0]
    edge2 = patch_points[3] - patch_points[0]
    patch_normal = np.cross(edge1, edge2)
    patch_normal = patch_normal / np.linalg.norm(patch_normal)

    # Ensure normal points toward receiver
    patch_center = np.mean(patch_points, axis=0)
    to_point = point - patch_center
    if np.dot(patch_normal, to_point) < 0:
        patch_normal = -patch_normal

    # Gauss-Legendre nodes/weights on [-1,1]
    u_nodes, u_wts = leggauss(N)
    v_nodes, v_wts = leggauss(N)
    # map nodes/weights to [0,1]
    u = 0.5 * (u_nodes + 1.0)
    v = 0.5 * (v_nodes + 1.0)
    wu = 0.5 * u_wts
    wv = 0.5 * v_wts

    # Create tensor product grid
    U, V = np.meshgrid(u, v, indexing='ij')       # shape (N, N)
    WU, WV = np.meshgrid(wu, wv, indexing='ij')   # shape (N, N)
    W = (WU * WV).ravel()                         # flattened weights

    Uf = U.ravel()
    Vf = V.ravel()

    p0, p1, p2, p3 = patch_points

    # bilinear interpolation weights
    w0 = (1.0 - Uf) * (1.0 - Vf)
    w1 = Uf * (1.0 - Vf)
    w2 = Uf * Vf
    w3 = (1.0 - Uf) * Vf

    # Compute patch sample points (Nsamples, 3)
    patch_pts = (w0[:, None] * p0[None, :]
                 + w1[:, None] * p1[None, :]
                 + w2[:, None] * p2[None, :]
                 + w3[:, None] * p3[None, :])

    # Vector from patch to point
    r_vec = point[None, :] - patch_pts           # (Nsamples,3)
    r_dist = np.linalg.norm(r_vec, axis=1)
    valid = r_dist > eps
    r_dist_safe = np.where(valid, r_dist, 1.0)   # avoid div-by-zero for masked entries
    r_dir = np.zeros_like(r_vec)
    r_dir[valid] = r_vec[valid] / r_dist_safe[valid, None]

    # cos(theta_i)
    cos_theta_i = np.dot(r_dir, patch_normal)

    # Jacobian (dP/du, dP/dv) depend on u,v -> vectorize
    dPdu = (-(1.0 - Vf)[:, None] * p0[None, :]
            + (1.0 - Vf)[:, None] * p1[None, :]
            + Vf[:, None] * p2[None, :]
            - Vf[:, None] * p3[None, :])

    dPdv = (-(1.0 - Uf)[:, None] * p0[None, :]
            - Uf[:, None] * p1[None, :]
            + Uf[:, None] * p2[None, :]
            + (1.0 - Uf)[:, None] * p3[None, :])

    cross = np.cross(dPdu, dPdv)
    dS = np.linalg.norm(cross, axis=1)

    # integrand per sample (mask invalid distances)
    integrand = np.zeros_like(r_dist)
    integrand[valid] = (1.0 / np.pi) * cos_theta_i[valid] * dS[valid] / (r_dist_safe[valid] ** 2)

    # Sum weighted integrand
    result = np.sum(integrand * W)

    if mode == 'source':
        result *= 0.25

    # No rigorous error estimate available for this deterministic quadrature
    return result, 0.0


patch_points = np.array([[-1.        , -1.        ,  0.        ],
                        [-0.33333333, -1.        ,  0.        ],
                        [-0.33333333, -0.33333333,  0.        ],
                        [-1.        , -0.33333333,  0.        ]])

src_point = np.array([0.70710678, 0.        , 0.70710678])

# Performance test
N = 50  # Number of repetitions (increase for more robust timing)

start = time.time()
for i in range(N):
    factor =pt_solution(src_point,patch_points)
end = time.time()
print("pt_solution")
print(f"Computed {N} factors in {end-start:.4f} seconds.")
print(f"Average time per calculation: {(end-start)/N:.6f} seconds.")

start = 0
end = 0
start = time.time()
for i in range(N):
    factor= point_patch_factor_leggaus(patch_points, src_point)
end = time.time()
print("point_patch_factor_leggaus")
print(f"Computed {N} factors in {end-start:.4f} seconds.")
print(f"Average time per calculation: {(end-start)/N:.6f} seconds.")

start = 0
end = 0
start = time.time()
for i in range(N):
    factor= point_patch_factor_leggaus_planar(patch_points, src_point)
end = time.time()
print("point_patch_factor_leggaus_planar")
print(f"Computed {N} factors in {end-start:.4f} seconds.")
print(f"Average time per calculation: {(end-start)/N:.6f} seconds.")

start = 0
end = 0
start = time.time()
for i in range(N):
    factor= point_patch_factor_dblquad(patch_points, src_point)
end = time.time()
print("point_patch_factor_dblquad")
print(f"Computed {N} factors in {end-start:.4f} seconds.")
print(f"Average time per calculation: {(end-start)/N:.6f} seconds.")

start = 0
end = 0
start = time.time()
for i in range(N):
    factor= point_patch_factor_mc_planar(patch_points, src_point)
end = time.time()
print("point_patch_factor_mc_planar")
print(f"Computed {N} factors in {end-start:.4f} seconds.")
print(f"Average time per calculation: {(end-start)/N:.6f} seconds.")

start = 0
end = 0
start = time.time()
for i in range(N):
    factor= point_patch_factor_dblquad_fix(patch_points, src_point)
end = time.time()
print("point_patch_factor_mc_planar")
print(f"Computed {N} factors in {end-start:.4f} seconds.")
print(f"Average time per calculation: {(end-start)/N:.6f} seconds.")
