"""
Kernel Integration for FK Green's Function Computation

This module implements the kernel computation and integration
over wavenumber space, following the FK algorithm.

The algorithm computes displacement kernels from a point source in a
multi-layered medium using the Haskell propagator matrix method with
compound matrix formulation for numerical stability.

References:
    Haskell (1964), BSSA
    Wang & Herrmann (1980), BSSA
    Zhu & Rivera (2002), GJI
"""

import numpy as np
from typing import Tuple, Dict
from scipy.special import j0, j1, jn

EPSILON = 1.0e-10
PI = np.pi
PI2 = 2.0 * PI


@jit(nopython=True, cache=True)
def bessel_functions(z: float) -> Tuple[float, float, float]:
    """
    Compute Bessel functions J0, J1, J2.

    For large arguments, uses asymptotic approximation.
    For small arguments, uses library functions (via fallback).

    Parameters
    ----------
    z : float
        Argument

    Returns
    -------
    tuple
        (J0(z), J1(z), J2(z))
    """
    if z < 0.01:
        # Small argument approximations
        z2 = z * z
        aj0 = 1.0 - z2 / 4.0
        aj1 = z / 2.0 * (1.0 - z2 / 8.0)
        aj2 = z2 / 8.0 * (1.0 - z2 / 12.0)
    elif z > 20.0:
        # Large argument asymptotic approximation
        phi = z - 0.25 * PI
        amp = 1.0 / np.sqrt(0.5 * PI * z)
        aj0 = np.cos(phi) * amp
        aj1 = np.sin(phi) * amp
        aj2 = -aj0  # Approximation
    else:
        # Use recurrence relation
        # J_{n+1}(z) = (2n/z) * J_n(z) - J_{n-1}(z)
        # This is a placeholder - actual implementation uses scipy
        phi = z - 0.25 * PI
        amp = 1.0 / np.sqrt(0.5 * PI * z)
        aj0 = np.cos(phi) * amp
        aj1 = np.sin(phi) * amp
        # J2 from recurrence: J2 = (2/z)*J1 - J0
        aj2 = (2.0 / z) * aj1 - aj0

    return aj0, aj1, aj2


def bessel_functions_scipy(z: float) -> Tuple[float, float, float]:
    """
    Compute Bessel functions using scipy (more accurate).

    Parameters
    ----------
    z : float
        Argument

    Returns
    -------
    tuple
        (J0(z), J1(z), J2(z))
    """
    return j0(z), j1(z), jn(2, z)


@jit(nopython=True, cache=True)
def compute_kernel(
    k: float,
    omega: complex,
    model_d: np.ndarray,
    model_vp: np.ndarray,
    model_vs: np.ndarray,
    model_rho: np.ndarray,
    model_qp: np.ndarray,
    model_qs: np.ndarray,
    model_mu: np.ndarray,
    model_xi: np.ndarray,
    n_layers: int,
    src_layer: int,
    rcv_layer: int,
    stype: int,
    updn: int,
    flip: int,
    si: np.ndarray,
) -> np.ndarray:
    """
    Compute displacement kernels U(k, omega) for all azimuthal modes.

    This is the core FK kernel computation following Haskell (1964)
    with compound matrix formulation for numerical stability.

    Parameters
    ----------
    k : float
        Wavenumber
    omega : complex
        Complex frequency (with damping)
    model_* : ndarray
        Model arrays (d, vp, vs, rho, qp, qs, mu, xi)
    n_layers : int
        Number of layers
    src_layer : int
        Source layer index (1-based)
    rcv_layer : int
        Receiver layer index (1-based)
    stype : int
        Source type
    updn : int
        Up/down selection
    flip : int
        Model flip
    si : ndarray
        Source coefficients

    Returns
    -------
    ndarray
        3x3 array of kernels u[n][component], n=0,1,2, component=Z,R,T
    """
    u = np.zeros((3, 3), dtype=np.complex128)

    # Initialize B matrix and g vector
    b = np.eye(7, dtype=np.complex128)
    g = np.zeros(7, dtype=np.complex128)
    e = np.zeros(7, dtype=np.complex128)
    z = np.zeros((3, 5), dtype=np.complex128)

    # Propagation from bottom to top
    for j in range(n_layers - 1, -1, -1):
        layer_idx = j

        # Get layer parameters
        ra, rb, r, r1, kd, mu2 = compute_layer_parameters(
            k,
            omega,
            model_vp[layer_idx],
            model_vs[layer_idx],
            model_mu[layer_idx],
            model_qp[layer_idx],
            model_qs[layer_idx],
            model_d[layer_idx],
        )

        if j == n_layers - 1 and model_d[layer_idx] < EPSILON:
            # Initialize g in bottom halfspace
            g = initial_g(ra, rb, r, r1, mu2)
        elif j == 0 and model_d[0] < EPSILON:
            # Top halfspace: compute e vector
            e = e_vector(ra, rb, r, r1, mu2)
            break
        else:
            # Compute compound matrix and propagate
            c, exa, exb, Ca, Cb, Ya, Yb, Xa, Xb = compound_matrix(
                ra, rb, r, r1, kd, mu2
            )
            g = propagate_g(c, g)

        # Source layer: initialize Z
        if j + 1 == src_layer:
            # Separate source if needed
            ss = separate_source(
                si.astype(np.complex128), stype, updn, ra, rb, r, r1, mu2
            )
            z = initial_z(ss, g)

        # Above source: propagate Z with Haskell matrix
        if j + 1 < src_layer:
            if j + 1 >= rcv_layer:
                c, exa, exb, Ca, Cb, Ya, Yb, Xa, Xb = compound_matrix(
                    ra, rb, r, r1, kd, mu2
                )
                a = haskell_matrix(
                    ra, rb, r, r1, kd, mu2, exa, exb, Ca, Cb, Ya, Yb, Xa, Xb
                )
                z = propagate_z(a, z)
            else:
                c, exa, exb, Ca, Cb, Ya, Yb, Xa, Xb = compound_matrix(
                    ra, rb, r, r1, kd, mu2
                )
                b = propagate_b(c, b)

    # Apply top boundary condition
    e[2] = 2.0 * e[2]
    rayl = g[0] * e[0] + g[1] * e[1] + g[2] * e[2] + g[3] * e[3] + g[4] * e[4]
    love = g[5] * e[5] + g[6] * e[6]

    # Apply B matrix to e
    g_new = np.zeros(7, dtype=np.complex128)
    for i in range(4):
        for j_idx in range(5):
            g_new[i] += b[i, j_idx] * e[j_idx]
    g_new[2] = g_new[2] / 2.0
    g_new[5] = b[5, 5] * e[5] + b[5, 6] * e[6]

    # Compute final kernels
    for i in range(3):
        dum = z[i, 1] * g_new[0] + z[i, 2] * g_new[1] - z[i, 3] * g_new[2]
        z[i, 1] = -z[i, 0] * g_new[0] + z[i, 2] * g_new[2] + z[i, 3] * g_new[3]
        z[i, 0] = dum
        z[i, 4] = z[i, 4] * g_new[5]

    # Scale factor
    dum = k + 0j
    if stype == 1:  # Single force
        dum = 1.0 + 0j

    # Displacement kernels
    for i in range(3):
        if np.abs(rayl) > EPSILON:
            u[i, 0] = dum * z[i, 1] / rayl  # Z
            u[i, 1] = dum * z[i, 0] / rayl  # R
        if np.abs(love) > EPSILON:
            u[i, 2] = dum * z[i, 4] / love  # T

    return u


def integrate_wavenumber(
    omega: complex,
    model_arrays: dict,
    n_layers: int,
    src_layer: int,
    rcv_layer: int,
    stype: int,
    updn: int,
    flip: int,
    si: np.ndarray,
    distances_km: np.ndarray,
    k_min: float,
    k_max: float,
    dk: float,
) -> np.ndarray:
    """
    Integrate over wavenumber to compute frequency-domain response.

    Parameters
    ----------
    omega : complex
        Complex frequency
    model_arrays : dict
        Model parameter arrays
    n_layers : int
        Number of layers
    src_layer, rcv_layer : int
        Source and receiver layer indices
    stype : int
        Source type
    updn : int
        Up/down selection
    flip : int
        Model flip
    si : ndarray
        Source coefficients
    distances_km : ndarray
        Array of distances in km
    k_min, k_max : float
        Wavenumber integration limits
    dk : float
        Wavenumber step

    Returns
    -------
    ndarray
        shape (n_distances, n_components, n_modes) frequency-domain response
    """
    n_dist = len(distances_km)
    n_com = 3 + 3 * stype  # Number of components based on source type

    # Initialize sum array
    # Layout: [n=0: Z,R,T], [n=1: Z,R,T], [n=2: Z,R,T] -> 9 total for DC
    sums = np.zeros((n_dist, 9), dtype=np.complex128)

    # Extract model arrays
    d = model_arrays["d"]
    vp = model_arrays["vp"]
    vs = model_arrays["vs"]
    rho = model_arrays["rho"]
    qp = model_arrays["qp"]
    qs = model_arrays["qs"]
    mu = model_arrays["mu"]
    xi = model_arrays["xi"]

    # Wavenumber integration
    k = k_min + 0.5 * dk
    n_k = int((k_max - k) / dk)

    for i_k in range(n_k):
        # Compute kernel
        u = compute_kernel(
            k,
            omega,
            d,
            vp,
            vs,
            rho,
            qp,
            qs,
            mu,
            xi,
            n_layers,
            src_layer,
            rcv_layer,
            stype,
            updn,
            flip,
            si,
        )

        # Sum over distances with Bessel function weighting
        for ix, x in enumerate(distances_km):
            z = k * x
            aj0, aj1, aj2 = bessel_functions_scipy(z)

            # n=0 mode
            sums[ix, 0] += u[0, 0] * aj0 * flip  # Z
            sums[ix, 1] -= u[0, 1] * aj1  # R
            sums[ix, 2] -= u[0, 2] * aj1  # T

            # n=1 mode
            if z > EPSILON:
                nf = (u[1, 1] + u[1, 2]) * aj1 / z
            else:
                nf = 0.0 + 0j
            sums[ix, 3] += u[1, 0] * aj1 * flip  # Z
            sums[ix, 4] += u[1, 1] * aj0 - nf  # R
            sums[ix, 5] += u[1, 2] * aj0 - nf  # T

            # n=2 mode
            if z > EPSILON:
                nf = 2.0 * (u[2, 1] + u[2, 2]) * aj2 / z
            else:
                nf = 0.0 + 0j
            sums[ix, 6] += u[2, 0] * aj2 * flip  # Z
            sums[ix, 7] += u[2, 1] * aj1 - nf  # R
            sums[ix, 8] += u[2, 2] * aj1 - nf  # T

        k += dk

    return sums


def compute_complex_wavenumbers(
    omega: complex, vp: float, vs: float, qp: float, qs: float
) -> Tuple[complex, complex]:
    """
    Compute complex wavenumbers with attenuation.

    Uses Futterman Q model following Aki & Richards p182.

    Parameters
    ----------
    omega : complex
        Complex frequency
    vp, vs : float
        P and S velocities
    qp, qs : float
        Quality factors

    Returns
    -------
    tuple
        (ka, kb) complex wavenumbers squared
    """
    att = np.log(omega / PI2) / PI + 0.5j

    ka = omega / (vp * (1.0 + att / qp))
    kb = omega / (vs * (1.0 + att / qs))

    return ka * ka, kb * kb
