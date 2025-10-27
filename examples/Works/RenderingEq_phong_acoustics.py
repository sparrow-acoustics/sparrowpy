# %%
import sparrowpy as sp
import numpy as np
import pyfar as pf
import sofar as sf
import matplotlib.pyplot as plt
import sys
import os
from numpy.polynomial.legendre import leggauss
import sparrowpy.geometry as geom

def dir_to_angles(v):
    phi = np.arctan2(v[:,1], v[:,0])          # [-pi, pi]
    theta = np.arccos(np.clip(v[:,2], -1, 1)) # [0, pi]
    return phi, theta

# -----------------------------
# Geometry helpers
# -----------------------------
def ortho_normal_basis(n):
    a = np.array([1.0, 0.0, 0.0]) if abs(n[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    t = np.cross(a, n); t /= (np.linalg.norm(t) + 1e-12)
    b = np.cross(n, t)
    return t, b

def rotate_tb(t, b, psi):
    c, s = np.cos(psi), np.sin(psi)
    tx = c*t + s*b
    ty = -s*t + c*b
    return tx, ty

def arvo_N(w, v, n):
    w = np.asarray(w, float); v = np.asarray(v, float)
    w /= np.linalg.norm(w); v /= np.linalg.norm(v)
    d = float(np.clip(np.dot(w, v), -1.0, 1.0))
    c = float(np.sqrt(max(0.0, 1.0 - d*d)))
    even = (n % 2 == 0)
    if even:
        t = np.pi * 0.5; A = np.pi * 0.5; i = 0
    else:
        t = c; A = np.pi - np.arccos(d); i = 1
    S = 0.0; T = t
    while i <= n - 2:
        S = S + T
        T = T * (c*c) * ((i + 1.0) / (i + 2.0))
        i += 2
    return 2.0 * (T + d * A + (d*d) * S) / (n + 2.0)

def acoustic_phong(dir_in, dir_out, normal, scattering_coef, phong_exponent=20):
    """
    Acoustic Phong BRDF model.

    Parameters
    ----------
    dir_in : array-like, shape (3,)
        Incoming direction (unit vector).
    dir_out : array-like, shape (3,)
        Outgoing direction (unit vector).
    normal : array-like, shape (3,), optional
        Surface normal (unit vector). Default is [0, 0, 1].
    phong_exponent : int, optional
        Phong exponent controlling the lobe sharpness. Default is 10.

    Returns
    -------
    fr : float
        BRDF value.
    """
    dir_in = dir_in / np.linalg.norm(dir_in)
    dir_out = dir_out / np.linalg.norm(dir_out)
    normal = normal / np.linalg.norm(normal)

    # Compute reflection direction
    reflect_dir = 2 * np.dot(dir_in, normal) * normal - dir_in
    reflect_dir = reflect_dir / np.linalg.norm(reflect_dir)
    if scattering_coef < 1:
        N = arvo_N(reflect_dir, normal, phong_exponent)
    else:
        N = 1.0
    if N == 0:
        return 0.0
    alpha = 0.0 #no absorption
    fr = (1 - scattering_coef)*(1-alpha)*abs(np.dot(reflect_dir, dir_out))**phong_exponent / N + scattering_coef*(1-alpha)/np.pi

    return fr

def integrate_patch(source_point, receiver_point, patch_points, patch_normal,
                        N=9, scattering_coef=1.0, phong_exponent=999, n_bins=1):
    """
    Integrate patch BRDF contribution from a point source to a receiver.
    Returns E_p (array of length n_bins).
    """
    p0 = patch_points[0]
    edge_u = patch_points[1] - p0
    edge_v = patch_points[3] - p0
    I=1/(4*np.pi)
    dS = np.linalg.norm(np.cross(edge_u, edge_v))

    u_nodes, u_weights = leggauss(N)
    v_nodes, v_weights = leggauss(N)

    E_p = np.zeros(n_bins, dtype=np.float64)

    for i in range(N):
        u = 0.5 * (u_nodes[i] + 1.0)
        wu = 0.5 * u_weights[i]

        for j in range(N):
            v = 0.5 * (v_nodes[j] + 1.0)
            wv = 0.5 * v_weights[j]

            xi = p0 + u * edge_u + v * edge_v

            vec_source_to_xi = source_point - xi
            dist_source_xi = np.linalg.norm(vec_source_to_xi)
            dir_source_to_xi = vec_source_to_xi / (dist_source_xi + 1e-12)

            cos_theta_in = max(0.0, np.dot(patch_normal, dir_source_to_xi))
            Li = I / (dist_source_xi**2)

            vec_xi_to_receiver = receiver_point - xi
            dist_xi_receiver = np.linalg.norm(vec_xi_to_receiver)
            dir_xi_to_receiver = vec_xi_to_receiver / (dist_xi_receiver + 1e-12)

            cos_theta_out_patch = max(0.0, np.dot(patch_normal, dir_xi_to_receiver))

            dA = dS * wu * wv
            V = 1.0

            geom_out = (cos_theta_out_patch) / (dist_xi_receiver ** 2)
            fr = acoustic_phong(dir_source_to_xi, dir_xi_to_receiver, patch_normal,
                                scattering_coef, phong_exponent=phong_exponent)

            Lo_val = Li * fr * cos_theta_in * geom_out * V * dA
            E_p += Lo_val

    return E_p

#%%

if __name__ == "__main__":
        
    incidence = 30
    incidence_rcv = 30
    incidence_rad = np.deg2rad(incidence)
    incidence_rcv_rad = np.deg2rad(incidence_rcv)
    distance_from_patch = 1
    source = pf.Coordinates.from_spherical_colatitude(0,incidence_rad,distance_from_patch)
    receiver = pf.Coordinates.from_spherical_colatitude(np.pi,incidence_rcv_rad,source.radius[0])
    
    width = 20
    length = 20
    patch_size = 20
    patch = sp.geometry.Polygon(
            [[-width/2, -length/2, 0],
                [width/2, -length/2, 0],
                [width/2, length/2, 0],
                [-width/2, length/2, 0]],
            [1, 0, 0], [0, 0, 1])
    patch_points = patch.pts
    normal = patch.normal
    
    # call the function and assign E_p for downstream code
    point = source.cartesian[0]
    receiver_point = receiver.cartesian[0]
    patch_normal = np.array([0,0,1])
    phong_exp = 20
    #numerical
    E_specular_num = integrate_patch(point, receiver_point, patch_points, patch_normal,
                          N=9, scattering_coef=0, phong_exponent=phong_exp)
    E_specular_num_highersampling = integrate_patch(point, receiver_point, patch_points, patch_normal,
                          N=110, scattering_coef=0, phong_exponent=phong_exp)
    E_specular_num_veryhighsampling = integrate_patch(point, receiver_point, patch_points, patch_normal,
                        N=220, scattering_coef=0, phong_exponent=phong_exp)
    E_specular_num_sampling_sohigh = integrate_patch(point, receiver_point, patch_points, patch_normal,
                    N=333, scattering_coef=0, phong_exponent=phong_exp)
    
    E_diffuse_num = integrate_patch(point, receiver_point, patch_points, patch_normal,
                          N=9, scattering_coef=1, phong_exponent=phong_exp)
    E_diffuse_num_highersampling = integrate_patch(point, receiver_point, patch_points, patch_normal,
                          N=110, scattering_coef=1, phong_exponent=phong_exp)
    E_diffuse_num_veryhighsampling = integrate_patch(point, receiver_point, patch_points, patch_normal,
                        N=220, scattering_coef=1, phong_exponent=phong_exp)
    E_diffuse_num_sampling_sohigh = integrate_patch(point, receiver_point, patch_points, patch_normal,
                    N=333, scattering_coef=1, phong_exponent=phong_exp)
    
    #analytical
    E_specular_analytical = 1/(4*np.pi*(source.radius + receiver.radius)**2)
    E_diffuse_analytical = np.cos(np.deg2rad(incidence_rcv)) *2* E_specular_analytical #The Lambert diffuse reflection model revisited, Svensson, U. Peter; Savioja, Lauri. #The Lambert diffuse reflection model revisited, Svensson, U. Peter; Savioja, Lauri -> the ratio F = E_diffuse / E_specular = 2*cos(incidence). Therefore analytical E_diffuse can be derived from analytical E_specular.
    
    print(f'analytical specular = {E_specular_analytical}')
    print(f'analytical diffuse = {E_diffuse_analytical}')

    Error_specular_num = abs(E_specular_num - E_specular_analytical)/E_specular_analytical * 100
    Error_specular_num_highersampling = abs(E_specular_num_highersampling - E_specular_analytical)/E_specular_analytical * 100
    Error_specular_num_veryhighsampling = abs(E_specular_num_veryhighsampling - E_specular_analytical)/E_specular_analytical * 100
    Error_specular_num_sampling_sohigh = abs(E_specular_num_sampling_sohigh - E_specular_analytical)/E_specular_analytical * 100
    print(f'specular numerical = {E_specular_num}, Error = {Error_specular_num} %')
    print(f'specular numerical higher sampling = {E_specular_num_highersampling}, Error = {Error_specular_num_highersampling} %')
    print(f'specular numerical very high sampling = {E_specular_num_veryhighsampling}, Error = {Error_specular_num_veryhighsampling} %')
    print(f'specular numerical sampling so high = {E_specular_num_sampling_sohigh}, Error = {Error_specular_num_sampling_sohigh} %')

    
    Error_diffuse_num = abs(E_diffuse_num - E_diffuse_analytical)/E_diffuse_analytical * 100
    Error_diffuse_num_highersampling = abs(E_diffuse_num_highersampling - E_diffuse_analytical)/E_diffuse_analytical * 100
    Error_diffuse_num_veryhighsampling = abs(E_diffuse_num_veryhighsampling - E_diffuse_analytical)/E_diffuse_analytical * 100
    Error_diffuse_num_sampling_sohigh = abs(E_diffuse_num_sampling_sohigh - E_diffuse_analytical)/E_diffuse_analytical * 100
    print(f'diffuse numerical = {E_diffuse_num}, Error = {Error_diffuse_num} %')
    print(f'diffuse numerical higher sampling = {E_diffuse_num_highersampling}, Error = {Error_diffuse_num_highersampling} %')
    print(f'diffuse numerical very high sampling = {E_diffuse_num_veryhighsampling}, Error = {Error_diffuse_num_veryhighsampling} %')
    print(f'diffuse numerical sampling so high = {E_diffuse_num_sampling_sohigh}, Error = {Error_diffuse_num_sampling_sohigh} %')


