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


def bessel_functions_scipy(z: float) -> Tuple[float, float, float]:
    """
    Compute Bessel functions using scipy (most accurate).

    Parameters
    ----------
    z : float
        Argument

    Returns
    -------
    tuple
        (J0(z), J1(z), J2(z))
    """
    if z < EPSILON:
        return 1.0, 0.0, 0.0
    return j0(z), j1(z), jn(2, z)


def sh_ch(a: complex, kd: float) -> Tuple[complex, complex, complex, float]:
    """
    Compute cosh(a*kd), sinh(a*kd)/a, sinh(a*kd)*a and the exponential factor.

    Multiplies results by exp(-Real(a*kd)) to suppress overflow.

    Parameters
    ----------
    a : complex
        Vertical slowness (ra or rb)
    kd : float
        k * layer_thickness

    Returns
    -------
    tuple
        (C, Y, X, ex) where:
        C = cosh(a*kd) * ex
        Y = sinh(a*kd)/a * ex
        X = sinh(a*kd)*a * ex
        ex = exp(-Real(a*kd))
    """
    y = kd * a
    r = y.real
    i = y.imag
    ex = np.exp(-r)

    y_half = 0.5 * complex(np.cos(i), np.sin(i))
    x_half = ex * ex * np.conj(y_half)

    C = y_half + x_half
    X = y_half - x_half
    Y = X / a if abs(a) > EPSILON else complex(kd, 0)
    X = X * a

    return C, Y, X, ex


def layer_parameters(
    k: float,
    omega: complex,
    vp: float,
    vs: float,
    rho: float,
    qp: float,
    qs: float,
    d: float,
    mu: float,
) -> Dict:
    """
    Compute layer parameters for FK computation.

    Parameters
    ----------
    k : float
        Wavenumber
    omega : complex
        Complex frequency
    vp, vs : float
        P and S velocities
    rho : float
        Density
    qp, qs : float
        Q factors
    d : float
        Layer thickness
    mu : float
        Shear modulus

    Returns
    -------
    dict
        Layer parameters: ra, rb, r, r1, kd, mu2
    """
    k2 = k * k

    # Complex velocities with attenuation (Futterman Q model)
    if abs(omega) > EPSILON:
        att = np.log(omega / PI2) / PI + 0.5j
        vp_c = vp * (1.0 + att / qp)
        vs_c = vs * (1.0 + att / qs)
    else:
        vp_c = vp
        vs_c = vs

    # Complex wavenumber squared
    ka = (omega / vp_c) ** 2
    kb = (omega / vs_c) ** 2

    # Normalized wavenumbers
    kka = ka / k2 if k2 > EPSILON else 0
    kkb = kb / k2 if k2 > EPSILON else 0

    # Vertical slownesses
    ra = np.sqrt(1.0 - kka + 0j)
    rb = np.sqrt(1.0 - kkb + 0j)

    # r = 2/kkb, r1 = 1 - 1/r = 1 - kkb/2
    r = 2.0 / kkb if abs(kkb) > EPSILON else 0j
    r1 = 1.0 - 1.0 / r if abs(r) > EPSILON else 0j

    kd = k * d
    mu2 = 2.0 * mu

    return {"ra": ra, "rb": rb, "r": r, "r1": r1, "kd": kd, "mu2": mu2}


def compound_matrix(
    ra: complex, rb: complex, r: complex, r1: complex, kd: float, mu2: float
) -> np.ndarray:
    """
    Compute the 7x7 compound matrix.

    The upper-left 5x5 is the compound matrix of the 4x4 P-SV Haskell matrix.
    The lower-right 2x2 is the SH part.

    Parameters
    ----------
    ra, rb : complex
        Vertical P and S slownesses
    r, r1 : complex
        Layer parameters
    kd : float
        k * layer_thickness
    mu2 : float
        2 * shear_modulus

    Returns
    -------
    ndarray
        7x7 compound matrix
    """
    Ca, Ya, Xa, exa = sh_ch(ra, kd)
    Cb, Yb, Xb, exb = sh_ch(rb, kd)

    CaCb = Ca * Cb
    CaYb = Ca * Yb
    CaXb = Ca * Xb
    XaCb = Xa * Cb
    XaXb = Xa * Xb
    YaCb = Ya * Cb
    YaYb = Ya * Yb

    ex = exa * exb
    r2 = r * r
    r3 = r1 * r1
    one = complex(1.0, 0.0)

    a = np.zeros((7, 7), dtype=np.complex128)

    # P-SV part (5x5), scaled by exa*exb
    a[0, 0] = ((one + r3) * CaCb - XaXb - r3 * YaYb - 2.0 * r1 * ex) * r2
    a[0, 1] = (XaCb - CaYb) * r / mu2
    a[0, 2] = ((one + r1) * (CaCb - ex) - XaXb - r1 * YaYb) * r2 / mu2
    a[0, 3] = (YaCb - CaXb) * r / mu2
    a[0, 4] = (2.0 * (CaCb - ex) - XaXb - YaYb) * r2 / (mu2 * mu2)

    a[1, 0] = (r3 * YaCb - CaXb) * r * mu2
    a[1, 1] = CaCb
    a[1, 2] = (r1 * YaCb - CaXb) * r
    a[1, 3] = -Ya * Xb
    a[1, 4] = a[0, 3]

    a[2, 0] = 2.0 * mu2 * r2 * (r1 * r3 * YaYb - (CaCb - ex) * (r3 + r1) + XaXb)
    a[2, 1] = 2.0 * r * (r1 * CaYb - XaCb)
    a[2, 2] = 2.0 * (CaCb - a[0, 0]) + ex
    a[2, 3] = -2.0 * a[1, 2]
    a[2, 4] = -2.0 * a[0, 2]

    a[3, 0] = mu2 * r * (XaCb - r3 * CaYb)
    a[3, 1] = -Xa * Yb
    a[3, 2] = -a[2, 1] / 2.0
    a[3, 3] = a[1, 1]
    a[3, 4] = a[0, 1]

    a[4, 0] = mu2 * mu2 * r2 * (2.0 * (CaCb - ex) * r3 - XaXb - r3 * r3 * YaYb)
    a[4, 1] = a[3, 0]
    a[4, 2] = -a[2, 0] / 2.0
    a[4, 3] = a[1, 0]
    a[4, 4] = a[0, 0]

    # SH part (2x2), scaled by exb
    a[5, 5] = Cb
    a[5, 6] = -2.0 * Yb / mu2
    a[6, 5] = -mu2 * Xb / 2.0
    a[6, 6] = Cb

    return a


def haskell_matrix(
    ra: complex, rb: complex, r: complex, r1: complex, kd: float, mu2: float
) -> np.ndarray:
    """
    Compute the 5x5 Haskell propagator matrix.

    The 4x4 P-SV part follows Haskell (1964) eq. 17.
    The (5,5) element is exb for SH scaling.

    Parameters
    ----------
    ra, rb : complex
        Vertical P and S slownesses
    r, r1 : complex
        Layer parameters
    kd : float
        k * layer_thickness
    mu2 : float
        2 * shear_modulus

    Returns
    -------
    ndarray
        5x5 Haskell matrix
    """
    Ca, Ya, Xa, exa = sh_ch(ra, kd)
    Cb, Yb, Xb, exb = sh_ch(rb, kd)

    # Scale by cross factors
    Ca = Ca * exb
    Xa = Xa * exb
    Ya = Ya * exb
    Cb = Cb * exa
    Yb = Yb * exa
    Xb = Xb * exa

    a = np.zeros((5, 5), dtype=np.complex128)

    # P-SV part (4x4), scaled by exa*exb
    a[0, 0] = r * (Ca - r1 * Cb)
    a[0, 1] = r * (r1 * Ya - Xb)
    a[0, 2] = (Cb - Ca) * r / mu2
    a[0, 3] = (Xb - Ya) * r / mu2

    a[1, 0] = r * (r1 * Yb - Xa)
    a[1, 1] = r * (Cb - r1 * Ca)
    a[1, 2] = (Xa - Yb) * r / mu2
    a[1, 3] = -a[0, 2]

    a[2, 0] = mu2 * r * r1 * (Ca - Cb)
    a[2, 1] = mu2 * r * (r1 * r1 * Ya - Xb)
    a[2, 2] = a[1, 1]
    a[2, 3] = -a[0, 1]

    a[3, 0] = mu2 * r * (r1 * r1 * Yb - Xa)
    a[3, 1] = -a[2, 0]
    a[3, 2] = -a[1, 0]
    a[3, 3] = a[0, 0]

    # SH scaling factor
    a[4, 4] = exb

    return a


def e_vector(
    ra: complex, rb: complex, r: complex, r1: complex, mu2: float
) -> np.ndarray:
    """
    Compute the E vector for top boundary condition.

    For P-SV: E|_(12)^(ij), ij=12, 13, 23, 24, 34
    For SH: first column of SH E matrix

    Parameters
    ----------
    ra, rb : complex
        Vertical slownesses
    r, r1 : complex
        Layer parameters
    mu2 : float
        2 * shear_modulus

    Returns
    -------
    ndarray
        7-element E vector
    """
    e = np.zeros(7, dtype=np.complex128)

    # P-SV: E|_(12)^(ij), ij=12, 13, 23, 24, 34
    # Following exactly haskell.f eVector() subroutine
    e[0] = ra * rb - 1.0
    e[1] = mu2 * rb * (1.0 - r1)
    e[2] = mu2 * (r1 - ra * rb)
    e[3] = mu2 * ra * (r1 - 1.0)
    e[4] = mu2 * mu2 * (ra * rb - r1 * r1)

    # SH: first column of SH E matrix
    # Following exactly haskell.f eVector() subroutine
    e[5] = -1.0
    e[6] = mu2 * rb / 2.0

    return e


def initial_g(
    ra: complex, rb: complex, r: complex, r1: complex, mu2: float
) -> np.ndarray:
    """
    Initialize g vector in the bottom halfspace.

    g contains inverse(E)|_{ij}^{12} for P-SV and 5th row of SH E^-1.
    Following exactly haskell.f initialG() subroutine.

    Parameters
    ----------
    ra, rb : complex
        Vertical slownesses
    r, r1 : complex
        Layer parameters
    mu2 : float
        2 * shear_modulus

    Returns
    -------
    ndarray
        7-element g vector
    """
    g = np.zeros(7, dtype=np.complex128)

    # P-SV: inverse(E)|_{ij}^{12}, ij=12,13,23,24,34
    # Following exactly haskell.f initialG() subroutine
    # See EQ 33 on ZR/p623, constant omitted
    delta = r * (1.0 - ra * rb) - 1.0
    g[0] = mu2 * (delta - r1)
    g[1] = ra
    g[2] = delta
    g[3] = -rb
    g[4] = (1.0 + delta) / mu2

    # SH: 5th row of E^-1, see EQ A4 on ZR/p625, 1/2 omitted
    g[5] = -1.0
    g[6] = 2.0 / (rb * mu2) if abs(rb) > EPSILON else 0j

    return g


def propagate_g(a: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Propagate g vector upward using compound matrix.

    g = g @ a (row vector times matrix)

    Parameters
    ----------
    a : ndarray
        7x7 compound matrix
    g : ndarray
        7-element g vector

    Returns
    -------
    ndarray
        Updated g vector
    """
    g_new = np.zeros(7, dtype=np.complex128)

    # P-SV
    for i in range(5):
        for j in range(5):
            g_new[i] += g[j] * a[j, i]

    # SH
    g_new[5] = g[5] * a[5, 5] + g[6] * a[6, 5]
    g_new[6] = g[5] * a[5, 6] + g[6] * a[6, 6]

    return g_new


def initial_z(s: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Initialize z vector at the source.

    z(j) = s(i) * X|_ij^12 for P-SV
    z(j) = s(i) * X(5,i) for SH

    where X is constructed from g.

    Parameters
    ----------
    s : ndarray
        (3, 6) source coefficients for n=0,1,2
    g : ndarray
        7-element g vector

    Returns
    -------
    ndarray
        (3, 5) z matrix
    """
    z = np.zeros((3, 5), dtype=np.complex128)

    for i in range(3):
        # P-SV: see Wang & Herrmann (1980) p1018
        z[i, 0] = -s[i, 1] * g[0] - s[i, 2] * g[1] + s[i, 3] * g[2]
        z[i, 1] = s[i, 0] * g[0] - s[i, 2] * g[2] - s[i, 3] * g[3]
        z[i, 2] = s[i, 0] * g[1] + s[i, 1] * g[2] - s[i, 3] * g[4]
        z[i, 3] = -s[i, 0] * g[2] + s[i, 1] * g[3] + s[i, 2] * g[4]
        # SH
        z[i, 4] = s[i, 4] * g[5] + s[i, 5] * g[6]

    return z


def propagate_z(a: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Propagate z vector using Haskell matrix.

    z = z @ a

    Parameters
    ----------
    a : ndarray
        5x5 Haskell matrix
    z : ndarray
        (3, 5) z matrix

    Returns
    -------
    ndarray
        Updated z matrix
    """
    z_new = np.zeros((3, 5), dtype=np.complex128)

    for i in range(3):
        # P-SV
        for j in range(4):
            for l in range(4):
                z_new[i, j] += z[i, l] * a[l, j]
        # SH: scaled by exb
        z_new[i, 4] = z[i, 4] * a[4, 4]

    return z_new


def propagate_b(c: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Propagate B matrix using compound matrix.

    b = b @ c

    Parameters
    ----------
    c : ndarray
        7x7 compound matrix
    b : ndarray
        7x7 B matrix

    Returns
    -------
    ndarray
        Updated B matrix
    """
    b_new = np.zeros((7, 7), dtype=np.complex128)

    # P-SV (5x5)
    for i in range(5):
        for j in range(5):
            for l in range(5):
                b_new[i, j] += b[i, l] * c[l, j]

    # SH (2x2)
    for i in range(5, 7):
        for j in range(5, 7):
            b_new[i, j] = b[i, 5] * c[5, j] + b[i, 6] * c[6, j]

    return b_new


def separate_source(
    si: np.ndarray,
    stype: int,
    updn: int,
    ra: complex,
    rb: complex,
    r: complex,
    r1: complex,
    mu2: float,
) -> np.ndarray:
    """
    Separate source into up-going and down-going waves.

    Parameters
    ----------
    si : ndarray
        Source coefficients
    stype : int
        Source type
    updn : int
        Up/down selection (0=all, 1=down, -1=up)
    ra, rb, r, r1 : complex
        Layer parameters
    mu2 : float
        2 * shear_modulus

    Returns
    -------
    ndarray
        Separated source coefficients
    """
    ii = stype + 1
    ss = np.zeros((3, 6), dtype=np.complex128)

    if updn == 0:
        # All waves
        for i in range(ii):
            for j in range(6):
                ss[i, j] = si[i, j]
        return ss

    # Down-going (updn=1) or up-going (updn=-1) matrix
    ra1 = 1.0 / ra if abs(ra) > EPSILON else 0
    rb1 = 1.0 / rb if abs(rb) > EPSILON else 0
    dum = updn * r

    temp = np.eye(4, dtype=np.complex128)
    temp[0, 1] = dum * (rb - r1 * ra1)
    temp[0, 3] = dum * (ra1 - rb) / mu2
    temp[1, 0] = dum * (ra - r1 * rb1)
    temp[1, 2] = dum * (rb1 - ra) / mu2
    temp[2, 1] = dum * (rb - r1 * r1 * ra1) * mu2
    temp[2, 3] = dum * (r1 * ra1 - rb)
    temp[3, 0] = dum * (ra - r1 * r1 * rb1) * mu2
    temp[3, 2] = dum * (r1 * rb1 - ra)

    temp_sh = (updn * 2.0 / mu2) * rb1

    for i in range(ii):
        for j in range(4):
            dum = 0j
            for jj in range(4):
                dum += temp[j, jj] * si[i, jj]
            ss[i, j] = dum / 2.0

        ss[i, 4] = (si[i, 4] + temp_sh * si[i, 5]) / 2.0
        if abs(temp_sh) > EPSILON:
            ss[i, 5] = (si[i, 5] + si[i, 4] / temp_sh) / 2.0
        else:
            ss[i, 5] = si[i, 5] / 2.0

    return ss


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
        Model arrays (d, vp, vs, rho, qp, qs, mu)
    n_layers : int
        Number of layers
    src_layer : int
        Source layer index (1-based)
    rcv_layer : int
        Receiver layer index (1-based)
    stype : int
        Source type (0=explosion, 1=force, 2=DC)
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

    # Initialize B as identity, e and g as unit vectors
    b = np.eye(7, dtype=np.complex128)
    e = np.zeros(7, dtype=np.complex128)
    g = np.zeros(7, dtype=np.complex128)

    # Default boundary conditions
    e[0] = 1.0
    e[5] = 1.0
    g[4] = 1.0
    g[6] = 1.0

    z = np.zeros((3, 5), dtype=np.complex128)

    # mb = n_layers (bottom layer index)
    mb = n_layers

    # Propagation from bottom to top
    for j in range(mb - 1, -1, -1):  # j = mb-1, ..., 0 (0-indexed)
        layer_idx = j

        # Get layer parameters
        params = layer_parameters(
            k,
            omega,
            model_vp[layer_idx],
            model_vs[layer_idx],
            model_rho[layer_idx],
            model_qp[layer_idx],
            model_qs[layer_idx],
            model_d[layer_idx],
            model_mu[layer_idx],
        )
        ra, rb = params["ra"], params["rb"]
        r, r1 = params["r"], params["r1"]
        kd, mu2 = params["kd"], params["mu2"]

        j_1based = j + 1  # Convert to 1-based for comparisons

        if j_1based == mb and model_d[layer_idx] < EPSILON:
            # Initialize g in bottom halfspace
            g = initial_g(ra, rb, r, r1, mu2)
        elif j_1based == 1 and model_d[0] < EPSILON:
            # Top halfspace: compute e vector
            e = e_vector(ra, rb, r, r1, mu2)
            break
        else:
            # Compute compound matrix and propagate
            c = compound_matrix(ra, rb, r, r1, kd, mu2)
            g = propagate_g(c, g)

        # Source layer: initialize Z
        if j_1based == src_layer:
            ss = separate_source(si, stype, updn, ra, rb, r, r1, mu2)
            z = initial_z(ss, g)

        # Above source: propagate Z with Haskell matrix
        if j_1based < src_layer:
            if j_1based >= rcv_layer:
                # Use Haskell matrix
                a = haskell_matrix(ra, rb, r, r1, kd, mu2)
                z = propagate_z(a, z)
            else:
                # Use compound matrix for B
                c = compound_matrix(ra, rb, r, r1, kd, mu2)
                b = propagate_b(c, b)

    # Apply top boundary condition
    e[2] = 2.0 * e[2]

    rayl = g[0] * e[0] + g[1] * e[1] + g[2] * e[2] + g[3] * e[3] + g[4] * e[4]
    love = g[5] * e[5] + g[6] * e[6]

    # Apply B matrix to e
    g_new = np.zeros(7, dtype=np.complex128)
    for i in range(4):
        for jj in range(5):
            g_new[i] += b[i, jj] * e[jj]
    g_new[2] = g_new[2] / 2.0
    g_new[5] = b[5, 5] * e[5] + b[5, 6] * e[6]

    # Compute final displacements
    for i in range(3):
        dum = z[i, 1] * g_new[0] + z[i, 2] * g_new[1] - z[i, 3] * g_new[2]
        z[i, 1] = -z[i, 0] * g_new[0] + z[i, 2] * g_new[2] + z[i, 3] * g_new[3]
        z[i, 0] = dum
        z[i, 4] = z[i, 4] * g_new[5]

    # Displacement kernels
    dum = complex(k, 0)
    if stype == 1:  # Single force
        dum = complex(1.0, 0)

    for i in range(3):
        if abs(rayl) > EPSILON:
            u[i, 0] = dum * z[i, 1] / rayl  # Z
            u[i, 1] = dum * z[i, 0] / rayl  # R
        if abs(love) > EPSILON:
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
        shape (n_distances, 9) frequency-domain response
    """
    n_dist = len(distances_km)

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
            z_arg = k * x
            aj0, aj1, aj2 = bessel_functions_scipy(z_arg)

            # n=0 mode
            sums[ix, 0] += u[0, 0] * aj0 * flip  # Z
            sums[ix, 1] -= u[0, 1] * aj1  # R
            sums[ix, 2] -= u[0, 2] * aj1  # T

            # n=1 mode
            if z_arg > EPSILON:
                nf = (u[1, 1] + u[1, 2]) * aj1 / z_arg
            else:
                nf = 0.0 + 0j
            sums[ix, 3] += u[1, 0] * aj1 * flip  # Z
            sums[ix, 4] += u[1, 1] * aj0 - nf  # R
            sums[ix, 5] += u[1, 2] * aj0 - nf  # T

            # n=2 mode
            if z_arg > EPSILON:
                nf = 2.0 * (u[2, 1] + u[2, 2]) * aj2 / z_arg
            else:
                nf = 0.0 + 0j
            sums[ix, 6] += u[2, 0] * aj2 * flip  # Z
            sums[ix, 7] += u[2, 1] * aj1 - nf  # R
            sums[ix, 8] += u[2, 2] * aj1 - nf  # T

        k += dk

    return sums
