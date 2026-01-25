"""
Haskell Propagator Matrix Implementation

This module implements the Thompson-Haskell propagator matrix method
for computing seismic wave propagation in layered media.

The implementation follows Haskell (1964), with modifications for
numerical stability using compound matrices as described in
Wang & Herrmann (1980) and Zhu & Rivera (2002).
"""

import numpy as np
from typing import Tuple
from numba import jit, complex128, float64, int64


# Constants
EPSILON = 1e-10
PI = np.pi
PI2 = 2.0 * np.pi


@jit(nopython=True, cache=True)
def sh_ch(a: complex, kd: float) -> Tuple[complex, complex, complex, float]:
    """
    Compute hyperbolic functions with overflow suppression.

    Computes:
        c = cosh(a*kd)
        y = sinh(a*kd)/a
        x = sinh(a*kd)*a
        ex = exp(-Real(a*kd))

    The results are multiplied by ex to suppress overflow.

    Parameters
    ----------
    a : complex
        Vertical slowness (ra or rb)
    kd : float
        wavenumber * layer thickness

    Returns
    -------
    tuple
        (c, y, x, ex) - cosh, sinh/a, sinh*a, exponential factor
    """
    y = kd * a
    r = y.real
    i = y.imag
    ex = np.exp(-r)

    # e^(i*imag) = cos(imag) + i*sin(imag)
    y_temp = 0.5 * (np.cos(i) + 1j * np.sin(i))
    x_temp = ex * ex * np.conj(y_temp)

    c = y_temp + x_temp  # cosh
    x = y_temp - x_temp  # sinh

    # Avoid division by zero
    if np.abs(a) > EPSILON:
        y = x / a  # sinh/a
        x = x * a  # sinh*a
    else:
        y = kd * (1.0 + 0j)  # Limit: sinh(x)/x -> 1 as x -> 0
        x = 0.0 + 0j

    return c, y, x, ex


@jit(nopython=True, cache=True)
def compute_layer_parameters(
    k: float,
    omega: complex,
    vp: float,
    vs: float,
    mu: float,
    qp: float,
    qs: float,
    d: float,
) -> Tuple:
    """
    Compute parameters for a single layer.

    Parameters
    ----------
    k : float
        Wavenumber
    omega : complex
        Complex frequency (with damping)
    vp, vs : float
        P and S wave velocities
    mu : float
        Shear modulus
    qp, qs : float
        Quality factors
    d : float
        Layer thickness

    Returns
    -------
    tuple
        Layer parameters (ra, rb, r, r1, kd, mu2)
    """
    # Complex wavenumbers with attenuation (Futterman Q operator)
    # Following A&R p182
    att = np.log(omega / PI2) / PI + 0.5j

    ka = omega / (vp * (1.0 + att / qp))
    kb = omega / (vs * (1.0 + att / qs))

    ka = ka * ka  # (omega/vp_complex)^2
    kb = kb * kb  # (omega/vs_complex)^2

    k2 = k * k

    # Avoid division by zero
    if k2 < EPSILON:
        k2 = EPSILON

    kka = ka / k2
    kkb = kb / k2

    # r = 2*k^2/kb^2 = 2/kkb
    r = 2.0 / kkb

    # Vertical slownesses
    ra = np.sqrt(1.0 - kka)  # sqrt(1 - (ka/k)^2)
    rb = np.sqrt(1.0 - kkb)  # sqrt(1 - (kb/k)^2)

    # Ensure correct branch cut (Im(ra,rb) should be negative for evanescent waves)
    if ra.imag > 0:
        ra = -ra
    if rb.imag > 0:
        rb = -rb

    r1 = 1.0 - 1.0 / r  # 1 - kb^2/(2*k^2)
    kd = k * d
    mu2 = 2.0 * mu

    return ra, rb, r, r1, kd, mu2


@jit(nopython=True, cache=True)
def haskell_matrix(
    ra: complex,
    rb: complex,
    r: complex,
    r1: complex,
    kd: float,
    mu2: float,
    exa: float,
    exb: float,
    Ca: complex,
    Cb: complex,
    Ya: complex,
    Yb: complex,
    Xa: complex,
    Xb: complex,
) -> np.ndarray:
    """
    Compute 4x4 P-SV Haskell propagator matrix.

    Following Haskell (1964), Eq 17 of Zhu & Rivera (2002).
    The matrix relates displacement-stress vectors at the top
    and bottom of a layer.

    Parameters
    ----------
    ra, rb : complex
        Vertical slownesses for P and S waves
    r, r1 : complex
        Auxiliary parameters
    kd : float
        k * d (wavenumber * thickness)
    mu2 : float
        2 * shear modulus
    exa, exb : float
        Exponential scaling factors
    Ca, Cb, Ya, Yb, Xa, Xb : complex
        Hyperbolic function values

    Returns
    -------
    ndarray
        5x5 matrix (4x4 P-SV + 1 SH element)
    """
    a = np.zeros((5, 5), dtype=np.complex128)

    # Scale by opposite exponential for numerical stability
    Ca_scaled = Ca * exb
    Xa_scaled = Xa * exb
    Ya_scaled = Ya * exb
    Cb_scaled = Cb * exa
    Yb_scaled = Yb * exa
    Xb_scaled = Xb * exa

    # P-SV matrix, scaled by exa*exb
    a[0, 0] = r * (Ca_scaled - r1 * Cb_scaled)
    a[0, 1] = r * (r1 * Ya_scaled - Xb_scaled)
    a[0, 2] = (Cb_scaled - Ca_scaled) * r / mu2
    a[0, 3] = (Xb_scaled - Ya_scaled) * r / mu2

    a[1, 0] = r * (r1 * Yb_scaled - Xa_scaled)
    a[1, 1] = r * (Cb_scaled - r1 * Ca_scaled)
    a[1, 2] = (Xa_scaled - Yb_scaled) * r / mu2
    a[1, 3] = -a[0, 2]

    a[2, 0] = mu2 * r * r1 * (Ca_scaled - Cb_scaled)
    a[2, 1] = mu2 * r * (r1 * r1 * Ya_scaled - Xb_scaled)
    a[2, 2] = a[1, 1]
    a[2, 3] = -a[0, 1]

    a[3, 0] = mu2 * r * (r1 * r1 * Yb_scaled - Xa_scaled)
    a[3, 1] = -a[2, 0]
    a[3, 2] = -a[1, 0]
    a[3, 3] = a[0, 0]

    # SH element (Haskell matrix not needed, replaced by exb)
    a[4, 4] = exb

    return a


@jit(nopython=True, cache=True)
def compound_matrix(
    ra: complex, rb: complex, r: complex, r1: complex, kd: float, mu2: float
) -> Tuple[np.ndarray, float, float]:
    """
    Compute compound matrix of the Haskell propagator.

    The compound matrix avoids numerical overflow in deep layers
    by using minors of the propagator matrix.

    Following Wang & Herrmann (1980), p1035.

    Parameters
    ----------
    ra, rb : complex
        Vertical slownesses
    r, r1 : complex
        Auxiliary parameters
    kd : float
        k * d (wavenumber * thickness)
    mu2 : float
        2 * shear modulus

    Returns
    -------
    tuple
        (compound_matrix, exa, exb) - 7x7 matrix and exponential factors
    """
    # Compute hyperbolic functions
    Ca, Ya, Xa, exa = sh_ch(ra, kd)
    Cb, Yb, Xb, exb = sh_ch(rb, kd)

    # Products of hyperbolic functions
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

    # Allocate compound matrix
    a = np.zeros((7, 7), dtype=np.complex128)

    # P-SV part, scaled by exa*exb
    a[0, 0] = ((1.0 + r3) * CaCb - XaXb - r3 * YaYb - 2.0 * r1 * ex) * r2
    a[0, 1] = (XaCb - CaYb) * r / mu2
    a[0, 2] = ((1.0 + r1) * (CaCb - ex) - XaXb - r1 * YaYb) * r2 / mu2
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

    # SH part, scaled by exb
    a[5, 5] = Cb
    a[5, 6] = -2.0 * Yb / mu2
    a[6, 5] = -mu2 * Xb / 2.0
    a[6, 6] = Cb

    return a, exa, exb, Ca, Cb, Ya, Yb, Xa, Xb


@jit(nopython=True, cache=True)
def e_vector(
    ra: complex, rb: complex, r: complex, r1: complex, mu2: float
) -> np.ndarray:
    """
    Compute the E vector for free surface boundary condition.

    Parameters
    ----------
    ra, rb : complex
        Vertical slownesses
    r, r1 : complex
        Auxiliary parameters
    mu2 : float
        2 * shear modulus

    Returns
    -------
    ndarray
        7-element vector
    """
    e = np.zeros(7, dtype=np.complex128)

    # P-SV part: E|_(12)^(ij), ij=12, 13, 23, 24, 34
    e[0] = ra * rb - 1.0
    e[1] = mu2 * rb * (1.0 - r1)
    e[2] = mu2 * (r1 - ra * rb)
    e[3] = mu2 * ra * (r1 - 1.0)
    e[4] = mu2 * mu2 * (ra * rb - r1 * r1)

    # SH part
    e[5] = -1.0
    e[6] = mu2 * rb / 2.0

    return e


@jit(nopython=True, cache=True)
def initial_g(
    ra: complex, rb: complex, r: complex, r1: complex, mu2: float
) -> np.ndarray:
    """
    Initialize the g vector for the bottom halfspace.

    The g vector represents the boundary condition at the
    bottom of the model (radiation condition).

    Parameters
    ----------
    ra, rb : complex
        Vertical slownesses
    r, r1 : complex
        Auxiliary parameters
    mu2 : float
        2 * shear modulus

    Returns
    -------
    ndarray
        7-element vector
    """
    g = np.zeros(7, dtype=np.complex128)

    # P-SV: inverse(E)|_{ij}^{12}
    delta = r * (1.0 - ra * rb) - 1.0
    g[0] = mu2 * (delta - r1)
    g[1] = ra
    g[2] = delta
    g[3] = -rb
    g[4] = (1.0 + delta) / mu2

    # SH: 5th row of E^-1
    g[5] = -1.0
    g[6] = 2.0 / (rb * mu2)

    return g


@jit(nopython=True, cache=True)
def propagate_g(c: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Propagate g vector through a layer using compound matrix.

    Parameters
    ----------
    c : ndarray
        7x7 compound matrix
    g : ndarray
        7-element g vector

    Returns
    -------
    ndarray
        Updated g vector
    """
    g_new = np.zeros(7, dtype=np.complex128)

    # P-SV part (5x5)
    for i in range(5):
        for j in range(5):
            g_new[i] += c[j, i] * g[j]

    # SH part (2x2)
    g_new[5] = c[5, 5] * g[5] + c[6, 5] * g[6]
    g_new[6] = c[5, 6] * g[5] + c[6, 6] * g[6]

    return g_new


@jit(nopython=True, cache=True)
def propagate_b(c: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Propagate B matrix through a layer.

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

    # P-SV part (5x5)
    for i in range(5):
        for j in range(5):
            for k in range(5):
                b_new[i, j] += c[i, k] * b[k, j]

    # SH part (2x2)
    for i in range(5, 7):
        for j in range(5, 7):
            for k in range(5, 7):
                b_new[i, j] += c[i, k] * b[k, j]

    return b_new


@jit(nopython=True, cache=True)
def propagate_z(a: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Propagate Z matrix through a layer using Haskell matrix.

    Parameters
    ----------
    a : ndarray
        5x5 Haskell matrix
    z : ndarray
        3x5 Z matrix

    Returns
    -------
    ndarray
        Updated Z matrix
    """
    z_new = np.zeros((3, 5), dtype=np.complex128)

    # P-SV part (4x4)
    for n in range(3):  # azimuthal modes
        for i in range(4):
            for j in range(4):
                z_new[n, i] += a[i, j] * z[n, j]
        # SH part
        z_new[n, 4] = a[4, 4] * z[n, 4]

    return z_new


@jit(nopython=True, cache=True)
def compute_source_coefficients(
    stype: int, xi: float, mu: float, flip: int
) -> np.ndarray:
    """
    Compute source displacement-stress discontinuity coefficients.

    Parameters
    ----------
    stype : int
        Source type: 0=explosion, 1=single force, 2=double couple
    xi : float
        vs²/vp²
    mu : float
        Shear modulus
    flip : int
        1 or -1 for model orientation

    Returns
    -------
    ndarray
        3x6 array of source coefficients for n=0,1,2 modes
    """
    s = np.zeros((3, 6), dtype=np.float64)

    if stype == 2:  # Double couple
        s[0, 1] = 2.0 * xi / mu
        s[0, 3] = 4.0 * xi - 3.0
        s[1, 0] = flip / mu
        s[1, 4] = -s[1, 0]
        s[2, 3] = 1.0
        s[2, 5] = -1.0
    elif stype == 0:  # Explosion
        s[0, 1] = xi / mu
        s[0, 3] = 2.0 * xi
    elif stype == 1:  # Single force (multiplied by k)
        s[0, 2] = -flip
        s[1, 3] = -1.0
        s[1, 5] = 1.0

    return s


@jit(nopython=True, cache=True)
def separate_source(
    s: np.ndarray,
    stype: int,
    updn: int,
    ra: complex,
    rb: complex,
    r: complex,
    r1: complex,
    mu2: float,
) -> np.ndarray:
    """
    Separate source into up-going and down-going components.

    Parameters
    ----------
    s : ndarray
        3x6 source coefficients
    stype : int
        Source type
    updn : int
        Direction: 0=both, 1=down, -1=up
    ra, rb : complex
        Vertical slownesses
    r, r1 : complex
        Auxiliary parameters
    mu2 : float
        2 * shear modulus

    Returns
    -------
    ndarray
        3x6 separated source coefficients
    """
    if updn == 0:
        return s.astype(np.complex128)

    ss = np.zeros((3, 6), dtype=np.complex128)

    # Compute separation matrix
    ra1 = 1.0 / ra
    rb1 = 1.0 / rb
    dum = updn * r

    temp = np.zeros((4, 4), dtype=np.complex128)
    temp[0, 0] = 1.0
    temp[0, 1] = dum * (rb - r1 * ra1)
    temp[0, 2] = 0.0
    temp[0, 3] = dum * (ra1 - rb) / mu2
    temp[1, 0] = dum * (ra - r1 * rb1)
    temp[1, 1] = 1.0
    temp[1, 2] = dum * (rb1 - ra) / mu2
    temp[1, 3] = 0.0
    temp[2, 0] = 0.0
    temp[2, 1] = dum * (rb - r1 * r1 * ra1) * mu2
    temp[2, 2] = 1.0
    temp[2, 3] = dum * (r1 * ra1 - rb)
    temp[3, 0] = dum * (ra - r1 * r1 * rb1) * mu2
    temp[3, 1] = 0.0
    temp[3, 2] = dum * (r1 * rb1 - ra)
    temp[3, 3] = 1.0

    temp_sh = (updn * 2.0 / mu2) * rb1

    ii = stype + 1
    for i in range(ii):
        for j in range(4):
            dum = 0.0 + 0j
            for jj in range(4):
                dum += temp[j, jj] * s[i, jj]
            ss[i, j] = dum / 2.0
        ss[i, 4] = (s[i, 4] + temp_sh * s[i, 5]) / 2.0
        ss[i, 5] = (s[i, 5] + s[i, 4] / temp_sh) / 2.0

    return ss


@jit(nopython=True, cache=True)
def initial_z(ss: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Initialize Z matrix from source and g vector.

    Parameters
    ----------
    ss : ndarray
        3x6 source coefficients
    g : ndarray
        7-element g vector

    Returns
    -------
    ndarray
        3x5 Z matrix
    """
    z = np.zeros((3, 5), dtype=np.complex128)

    for n in range(3):  # azimuthal modes
        # P-SV: z(n,j) = s(n)*X|_{ij}^{12}
        z[n, 0] = ss[n, 1] * g[0] - ss[n, 3] * g[2]
        z[n, 1] = ss[n, 0] * g[0] - ss[n, 1] * g[1] - ss[n, 2] * g[2] + ss[n, 3] * g[3]
        z[n, 2] = ss[n, 0] * g[1] - ss[n, 2] * g[3]
        z[n, 3] = ss[n, 0] * g[2] - ss[n, 1] * g[3] - ss[n, 2] * g[4] + ss[n, 3] * g[1]
        # SH: z(n,5) = s(n)*X_5i
        z[n, 4] = ss[n, 4] * g[5] + ss[n, 5] * g[6]

    return z


@jit(nopython=True, cache=True)
def initialize_b() -> np.ndarray:
    """Initialize B matrix as identity."""
    b = np.eye(7, dtype=np.complex128)
    return b
