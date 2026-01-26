"""
Numba-optimized Kernel Integration for FK Green's Function Computation

This module implements the kernel computation with Numba JIT compilation
for high performance.
"""

import numpy as np
from numba import jit, prange
from numba.types import complex128, float64, int64
from scipy.special import j0, j1, jn

EPSILON = 1.0e-10
PI = np.pi
PI2 = 2.0 * PI


@jit(nopython=True, cache=True)
def sh_ch(a_real, a_imag, kd):
    """
    Compute cosh(a*kd), sinh(a*kd)/a, sinh(a*kd)*a and exp factor.
    Returns (C_real, C_imag, Y_real, Y_imag, X_real, X_imag, ex)
    """
    # y = kd * a
    y_real = kd * a_real
    y_imag = kd * a_imag

    # ex = exp(-real(y))
    ex = np.exp(-y_real)

    # y_half = 0.5 * exp(i*imag(y))
    yh_real = 0.5 * np.cos(y_imag)
    yh_imag = 0.5 * np.sin(y_imag)

    # x_half = ex^2 * conj(y_half)
    ex2 = ex * ex
    xh_real = ex2 * yh_real
    xh_imag = -ex2 * yh_imag

    # C = y_half + x_half (cosh)
    C_real = yh_real + xh_real
    C_imag = yh_imag + xh_imag

    # X_temp = y_half - x_half (sinh before scaling)
    Xt_real = yh_real - xh_real
    Xt_imag = yh_imag - xh_imag

    # Y = X_temp / a (sinh/a)
    a_mag2 = a_real * a_real + a_imag * a_imag
    if a_mag2 > EPSILON * EPSILON:
        Y_real = (Xt_real * a_real + Xt_imag * a_imag) / a_mag2
        Y_imag = (Xt_imag * a_real - Xt_real * a_imag) / a_mag2
    else:
        Y_real = kd
        Y_imag = 0.0

    # X = X_temp * a (sinh*a)
    X_real = Xt_real * a_real - Xt_imag * a_imag
    X_imag = Xt_real * a_imag + Xt_imag * a_real

    return C_real, C_imag, Y_real, Y_imag, X_real, X_imag, ex


@jit(nopython=True, cache=True)
def compute_layer_params(k, omega_real, omega_imag, vp, vs, mu, qp, qs, d):
    """
    Compute layer parameters.
    Returns (ra_r, ra_i, rb_r, rb_i, r_r, r_i, r1_r, r1_i, kd, mu2)
    """
    k2 = k * k
    if k2 < EPSILON:
        k2 = EPSILON

    # Complex frequency with attenuation
    if abs(omega_real) > EPSILON:
        # att = log(omega/pi2)/pi + 0.5j
        omega_mag = np.sqrt(omega_real * omega_real + omega_imag * omega_imag)
        att_real = np.log(omega_mag / PI2) / PI
        att_imag = 0.5

        # vp_c = vp * (1 + att/qp)
        vp_c_real = vp * (1.0 + att_real / qp)
        vp_c_imag = vp * att_imag / qp

        # vs_c = vs * (1 + att/qs)
        vs_c_real = vs * (1.0 + att_real / qs)
        vs_c_imag = vs * att_imag / qs
    else:
        vp_c_real = vp
        vp_c_imag = 0.0
        vs_c_real = vs
        vs_c_imag = 0.0

    # ka = (omega / vp_c)^2
    vp_mag2 = vp_c_real * vp_c_real + vp_c_imag * vp_c_imag
    div_r = (omega_real * vp_c_real + omega_imag * vp_c_imag) / vp_mag2
    div_i = (omega_imag * vp_c_real - omega_real * vp_c_imag) / vp_mag2
    ka_real = div_r * div_r - div_i * div_i
    ka_imag = 2.0 * div_r * div_i

    # kb = (omega / vs_c)^2
    vs_mag2 = vs_c_real * vs_c_real + vs_c_imag * vs_c_imag
    div_r = (omega_real * vs_c_real + omega_imag * vs_c_imag) / vs_mag2
    div_i = (omega_imag * vs_c_real - omega_real * vs_c_imag) / vs_mag2
    kb_real = div_r * div_r - div_i * div_i
    kb_imag = 2.0 * div_r * div_i

    # kka = ka / k2, kkb = kb / k2
    kka_real = ka_real / k2
    kka_imag = ka_imag / k2
    kkb_real = kb_real / k2
    kkb_imag = kb_imag / k2

    # ra = sqrt(1 - kka)
    tmp_real = 1.0 - kka_real
    tmp_imag = -kka_imag
    ra_real, ra_imag = csqrt(tmp_real, tmp_imag)

    # rb = sqrt(1 - kkb)
    tmp_real = 1.0 - kkb_real
    tmp_imag = -kkb_imag
    rb_real, rb_imag = csqrt(tmp_real, tmp_imag)

    # r = 2 / kkb
    kkb_mag2 = kkb_real * kkb_real + kkb_imag * kkb_imag
    if kkb_mag2 > EPSILON * EPSILON:
        r_real = 2.0 * kkb_real / kkb_mag2
        r_imag = -2.0 * kkb_imag / kkb_mag2
    else:
        r_real = 0.0
        r_imag = 0.0

    # r1 = 1 - 1/r
    r_mag2 = r_real * r_real + r_imag * r_imag
    if r_mag2 > EPSILON * EPSILON:
        r1_real = 1.0 - r_real / r_mag2
        r1_imag = r_imag / r_mag2
    else:
        r1_real = 0.0
        r1_imag = 0.0

    kd = k * d
    mu2 = 2.0 * mu

    return ra_real, ra_imag, rb_real, rb_imag, r_real, r_imag, r1_real, r1_imag, kd, mu2


@jit(nopython=True, cache=True)
def csqrt(x_real, x_imag):
    """Complex square root."""
    mag = np.sqrt(x_real * x_real + x_imag * x_imag)
    if mag < EPSILON:
        return 0.0, 0.0

    r = np.sqrt((mag + x_real) / 2.0)
    i = np.sqrt((mag - x_real) / 2.0)
    if x_imag < 0:
        i = -i
    return r, i


@jit(nopython=True, cache=True)
def cmul(a_r, a_i, b_r, b_i):
    """Complex multiplication."""
    return a_r * b_r - a_i * b_i, a_r * b_i + a_i * b_r


@jit(nopython=True, cache=True)
def cdiv(a_r, a_i, b_r, b_i):
    """Complex division."""
    denom = b_r * b_r + b_i * b_i
    if denom < EPSILON * EPSILON:
        return 0.0, 0.0
    return (a_r * b_r + a_i * b_i) / denom, (a_i * b_r - a_r * b_i) / denom


@jit(nopython=True, cache=True)
def compute_compound_matrix(ra_r, ra_i, rb_r, rb_i, r_r, r_i, r1_r, r1_i, kd, mu2, a):
    """
    Compute 7x7 compound matrix in-place.
    a is a pre-allocated (7,7) complex array.
    """
    # Get hyperbolic functions
    Ca_r, Ca_i, Ya_r, Ya_i, Xa_r, Xa_i, exa = sh_ch(ra_r, ra_i, kd)
    Cb_r, Cb_i, Yb_r, Yb_i, Xb_r, Xb_i, exb = sh_ch(rb_r, rb_i, kd)

    # Products
    CaCb_r, CaCb_i = cmul(Ca_r, Ca_i, Cb_r, Cb_i)
    CaYb_r, CaYb_i = cmul(Ca_r, Ca_i, Yb_r, Yb_i)
    CaXb_r, CaXb_i = cmul(Ca_r, Ca_i, Xb_r, Xb_i)
    XaCb_r, XaCb_i = cmul(Xa_r, Xa_i, Cb_r, Cb_i)
    XaXb_r, XaXb_i = cmul(Xa_r, Xa_i, Xb_r, Xb_i)
    YaCb_r, YaCb_i = cmul(Ya_r, Ya_i, Cb_r, Cb_i)
    YaYb_r, YaYb_i = cmul(Ya_r, Ya_i, Yb_r, Yb_i)
    YaXb_r, YaXb_i = cmul(Ya_r, Ya_i, Xb_r, Xb_i)
    XaYb_r, XaYb_i = cmul(Xa_r, Xa_i, Yb_r, Yb_i)

    ex = exa * exb
    r2_r, r2_i = cmul(r_r, r_i, r_r, r_i)
    r3_r, r3_i = cmul(r1_r, r1_i, r1_r, r1_i)

    # a[0,0] = ((1+r3)*CaCb - XaXb - r3*YaYb - 2*r1*ex) * r2
    t1_r = (1.0 + r3_r) * CaCb_r - r3_i * CaCb_i
    t1_i = (1.0 + r3_r) * CaCb_i + r3_i * CaCb_r
    t1_r -= XaXb_r
    t1_i -= XaXb_i
    t2_r, t2_i = cmul(r3_r, r3_i, YaYb_r, YaYb_i)
    t1_r -= t2_r
    t1_i -= t2_i
    t1_r -= 2.0 * r1_r * ex
    t1_i -= 2.0 * r1_i * ex
    a[0, 0] = complex(*cmul(t1_r, t1_i, r2_r, r2_i))

    # a[0,1] = (XaCb - CaYb) * r / mu2
    t1_r = XaCb_r - CaYb_r
    t1_i = XaCb_i - CaYb_i
    t2_r, t2_i = cmul(t1_r, t1_i, r_r, r_i)
    a[0, 1] = complex(t2_r / mu2, t2_i / mu2)

    # a[0,2] = ((1+r1)*(CaCb-ex) - XaXb - r1*YaYb) * r2 / mu2
    t1_r = (1.0 + r1_r) * (CaCb_r - ex) - r1_i * CaCb_i
    t1_i = (1.0 + r1_r) * CaCb_i + r1_i * (CaCb_r - ex)
    t1_r -= XaXb_r
    t1_i -= XaXb_i
    t2_r, t2_i = cmul(r1_r, r1_i, YaYb_r, YaYb_i)
    t1_r -= t2_r
    t1_i -= t2_i
    t2_r, t2_i = cmul(t1_r, t1_i, r2_r, r2_i)
    a[0, 2] = complex(t2_r / mu2, t2_i / mu2)

    # a[0,3] = (YaCb - CaXb) * r / mu2
    t1_r = YaCb_r - CaXb_r
    t1_i = YaCb_i - CaXb_i
    t2_r, t2_i = cmul(t1_r, t1_i, r_r, r_i)
    a[0, 3] = complex(t2_r / mu2, t2_i / mu2)

    # a[0,4] = (2*(CaCb-ex) - XaXb - YaYb) * r2 / (mu2*mu2)
    t1_r = 2.0 * (CaCb_r - ex) - XaXb_r - YaYb_r
    t1_i = 2.0 * CaCb_i - XaXb_i - YaYb_i
    t2_r, t2_i = cmul(t1_r, t1_i, r2_r, r2_i)
    mu4 = mu2 * mu2
    a[0, 4] = complex(t2_r / mu4, t2_i / mu4)

    # a[1,0] = (r3*YaCb - CaXb) * r * mu2
    t1_r, t1_i = cmul(r3_r, r3_i, YaCb_r, YaCb_i)
    t1_r -= CaXb_r
    t1_i -= CaXb_i
    t2_r, t2_i = cmul(t1_r, t1_i, r_r, r_i)
    a[1, 0] = complex(t2_r * mu2, t2_i * mu2)

    # a[1,1] = CaCb
    a[1, 1] = complex(CaCb_r, CaCb_i)

    # a[1,2] = (r1*YaCb - CaXb) * r
    t1_r, t1_i = cmul(r1_r, r1_i, YaCb_r, YaCb_i)
    t1_r -= CaXb_r
    t1_i -= CaXb_i
    a[1, 2] = complex(*cmul(t1_r, t1_i, r_r, r_i))

    # a[1,3] = -Ya*Xb
    a[1, 3] = complex(-YaXb_r, -YaXb_i)

    # a[1,4] = a[0,3]
    a[1, 4] = a[0, 3]

    # a[2,0] = 2*mu2*r2*(r1*r3*YaYb - (CaCb-ex)*(r3+r1) + XaXb)
    t1_r, t1_i = cmul(r1_r, r1_i, r3_r, r3_i)
    t2_r, t2_i = cmul(t1_r, t1_i, YaYb_r, YaYb_i)
    t3_r = r3_r + r1_r
    t3_i = r3_i + r1_i
    t4_r, t4_i = cmul(CaCb_r - ex, CaCb_i, t3_r, t3_i)
    t2_r -= t4_r
    t2_i -= t4_i
    t2_r += XaXb_r
    t2_i += XaXb_i
    t3_r, t3_i = cmul(t2_r, t2_i, r2_r, r2_i)
    a[2, 0] = complex(2.0 * mu2 * t3_r, 2.0 * mu2 * t3_i)

    # a[2,1] = 2*r*(r1*CaYb - XaCb)
    t1_r, t1_i = cmul(r1_r, r1_i, CaYb_r, CaYb_i)
    t1_r -= XaCb_r
    t1_i -= XaCb_i
    t2_r, t2_i = cmul(t1_r, t1_i, r_r, r_i)
    a[2, 1] = complex(2.0 * t2_r, 2.0 * t2_i)

    # a[2,2] = 2*(CaCb - a[0,0]) + ex
    a00 = a[0, 0]
    a[2, 2] = complex(2.0 * (CaCb_r - a00.real) + ex, 2.0 * (CaCb_i - a00.imag))

    # a[2,3] = -2*a[1,2]
    a12 = a[1, 2]
    a[2, 3] = complex(-2.0 * a12.real, -2.0 * a12.imag)

    # a[2,4] = -2*a[0,2]
    a02 = a[0, 2]
    a[2, 4] = complex(-2.0 * a02.real, -2.0 * a02.imag)

    # a[3,0] = mu2*r*(XaCb - r3*CaYb)
    t1_r, t1_i = cmul(r3_r, r3_i, CaYb_r, CaYb_i)
    t2_r = XaCb_r - t1_r
    t2_i = XaCb_i - t1_i
    t3_r, t3_i = cmul(t2_r, t2_i, r_r, r_i)
    a[3, 0] = complex(mu2 * t3_r, mu2 * t3_i)

    # a[3,1] = -Xa*Yb
    a[3, 1] = complex(-XaYb_r, -XaYb_i)

    # a[3,2] = -a[2,1]/2
    a21 = a[2, 1]
    a[3, 2] = complex(-a21.real / 2.0, -a21.imag / 2.0)

    # a[3,3] = a[1,1]
    a[3, 3] = a[1, 1]

    # a[3,4] = a[0,1]
    a[3, 4] = a[0, 1]

    # a[4,0] = mu2*mu2*r2*(2*(CaCb-ex)*r3 - XaXb - r3*r3*YaYb)
    t1_r, t1_i = cmul(r3_r, r3_i, CaCb_r - ex, CaCb_i)
    t1_r = 2.0 * t1_r - XaXb_r
    t1_i = 2.0 * t1_i - XaXb_i
    t2_r, t2_i = cmul(r3_r, r3_i, r3_r, r3_i)
    t3_r, t3_i = cmul(t2_r, t2_i, YaYb_r, YaYb_i)
    t1_r -= t3_r
    t1_i -= t3_i
    t2_r, t2_i = cmul(t1_r, t1_i, r2_r, r2_i)
    a[4, 0] = complex(mu4 * t2_r, mu4 * t2_i)

    # a[4,1] = a[3,0]
    a[4, 1] = a[3, 0]

    # a[4,2] = -a[2,0]/2
    a20 = a[2, 0]
    a[4, 2] = complex(-a20.real / 2.0, -a20.imag / 2.0)

    # a[4,3] = a[1,0]
    a[4, 3] = a[1, 0]

    # a[4,4] = a[0,0]
    a[4, 4] = a[0, 0]

    # SH part
    a[5, 5] = complex(Cb_r, Cb_i)
    a[5, 6] = complex(-2.0 * Yb_r / mu2, -2.0 * Yb_i / mu2)
    a[6, 5] = complex(-mu2 * Xb_r / 2.0, -mu2 * Xb_i / 2.0)
    a[6, 6] = complex(Cb_r, Cb_i)

    return exa, exb


@jit(nopython=True, cache=True)
def compute_haskell_matrix(ra_r, ra_i, rb_r, rb_i, r_r, r_i, r1_r, r1_i, kd, mu2, a):
    """
    Compute 5x5 Haskell matrix in-place.
    """
    Ca_r, Ca_i, Ya_r, Ya_i, Xa_r, Xa_i, exa = sh_ch(ra_r, ra_i, kd)
    Cb_r, Cb_i, Yb_r, Yb_i, Xb_r, Xb_i, exb = sh_ch(rb_r, rb_i, kd)

    # Scale by cross factors
    Ca_r, Ca_i = Ca_r * exb, Ca_i * exb
    Xa_r, Xa_i = Xa_r * exb, Xa_i * exb
    Ya_r, Ya_i = Ya_r * exb, Ya_i * exb
    Cb_r, Cb_i = Cb_r * exa, Cb_i * exa
    Yb_r, Yb_i = Yb_r * exa, Yb_i * exa
    Xb_r, Xb_i = Xb_r * exa, Xb_i * exa

    # a[0,0] = r * (Ca - r1*Cb)
    t1_r, t1_i = cmul(r1_r, r1_i, Cb_r, Cb_i)
    t2_r = Ca_r - t1_r
    t2_i = Ca_i - t1_i
    a[0, 0] = complex(*cmul(r_r, r_i, t2_r, t2_i))

    # a[0,1] = r * (r1*Ya - Xb)
    t1_r, t1_i = cmul(r1_r, r1_i, Ya_r, Ya_i)
    t2_r = t1_r - Xb_r
    t2_i = t1_i - Xb_i
    a[0, 1] = complex(*cmul(r_r, r_i, t2_r, t2_i))

    # a[0,2] = (Cb - Ca) * r / mu2
    t1_r = Cb_r - Ca_r
    t1_i = Cb_i - Ca_i
    t2_r, t2_i = cmul(t1_r, t1_i, r_r, r_i)
    a[0, 2] = complex(t2_r / mu2, t2_i / mu2)

    # a[0,3] = (Xb - Ya) * r / mu2
    t1_r = Xb_r - Ya_r
    t1_i = Xb_i - Ya_i
    t2_r, t2_i = cmul(t1_r, t1_i, r_r, r_i)
    a[0, 3] = complex(t2_r / mu2, t2_i / mu2)

    # a[1,0] = r * (r1*Yb - Xa)
    t1_r, t1_i = cmul(r1_r, r1_i, Yb_r, Yb_i)
    t2_r = t1_r - Xa_r
    t2_i = t1_i - Xa_i
    a[1, 0] = complex(*cmul(r_r, r_i, t2_r, t2_i))

    # a[1,1] = r * (Cb - r1*Ca)
    t1_r, t1_i = cmul(r1_r, r1_i, Ca_r, Ca_i)
    t2_r = Cb_r - t1_r
    t2_i = Cb_i - t1_i
    a[1, 1] = complex(*cmul(r_r, r_i, t2_r, t2_i))

    # a[1,2] = (Xa - Yb) * r / mu2
    t1_r = Xa_r - Yb_r
    t1_i = Xa_i - Yb_i
    t2_r, t2_i = cmul(t1_r, t1_i, r_r, r_i)
    a[1, 2] = complex(t2_r / mu2, t2_i / mu2)

    # a[1,3] = -a[0,2]
    a[1, 3] = -a[0, 2]

    # a[2,0] = mu2 * r * r1 * (Ca - Cb)
    t1_r = Ca_r - Cb_r
    t1_i = Ca_i - Cb_i
    t2_r, t2_i = cmul(r_r, r_i, r1_r, r1_i)
    t3_r, t3_i = cmul(t2_r, t2_i, t1_r, t1_i)
    a[2, 0] = complex(mu2 * t3_r, mu2 * t3_i)

    # a[2,1] = mu2 * r * (r1*r1*Ya - Xb)
    t1_r, t1_i = cmul(r1_r, r1_i, r1_r, r1_i)
    t2_r, t2_i = cmul(t1_r, t1_i, Ya_r, Ya_i)
    t2_r -= Xb_r
    t2_i -= Xb_i
    t3_r, t3_i = cmul(r_r, r_i, t2_r, t2_i)
    a[2, 1] = complex(mu2 * t3_r, mu2 * t3_i)

    # a[2,2] = a[1,1]
    a[2, 2] = a[1, 1]

    # a[2,3] = -a[0,1]
    a[2, 3] = -a[0, 1]

    # a[3,0] = mu2 * r * (r1*r1*Yb - Xa)
    t1_r, t1_i = cmul(r1_r, r1_i, r1_r, r1_i)
    t2_r, t2_i = cmul(t1_r, t1_i, Yb_r, Yb_i)
    t2_r -= Xa_r
    t2_i -= Xa_i
    t3_r, t3_i = cmul(r_r, r_i, t2_r, t2_i)
    a[3, 0] = complex(mu2 * t3_r, mu2 * t3_i)

    # a[3,1] = -a[2,0]
    a[3, 1] = -a[2, 0]

    # a[3,2] = -a[1,0]
    a[3, 2] = -a[1, 0]

    # a[3,3] = a[0,0]
    a[3, 3] = a[0, 0]

    # SH scaling
    a[4, 4] = complex(exb, 0.0)


@jit(nopython=True, cache=True)
def initial_g(ra_r, ra_i, rb_r, rb_i, r_r, r_i, r1_r, r1_i, mu2, g):
    """Initialize g vector in bottom halfspace - exactly matching Fortran."""
    # delta = r*(1-ra*rb) - 1
    rarb_r, rarb_i = cmul(ra_r, ra_i, rb_r, rb_i)
    t1_r = 1.0 - rarb_r
    t1_i = -rarb_i
    delta_r, delta_i = cmul(r_r, r_i, t1_r, t1_i)
    delta_r -= 1.0

    # g(1) = mu2*(delta - r1)
    g[0] = complex(mu2 * (delta_r - r1_r), mu2 * (delta_i - r1_i))

    # g(2) = ra
    g[1] = complex(ra_r, ra_i)

    # g(3) = delta
    g[2] = complex(delta_r, delta_i)

    # g(4) = -rb
    g[3] = complex(-rb_r, -rb_i)

    # g(5) = (1+delta)/mu2
    g[4] = complex((1.0 + delta_r) / mu2, delta_i / mu2)

    # g(6) = -1
    g[5] = complex(-1.0, 0.0)

    # g(7) = 2/(rb*mu2)
    rb_mag2 = rb_r * rb_r + rb_i * rb_i
    if rb_mag2 > EPSILON * EPSILON:
        g[6] = complex(2.0 * rb_r / (rb_mag2 * mu2), -2.0 * rb_i / (rb_mag2 * mu2))
    else:
        g[6] = complex(0.0, 0.0)


@jit(nopython=True, cache=True)
def e_vector(ra_r, ra_i, rb_r, rb_i, r_r, r_i, r1_r, r1_i, mu2, e):
    """Compute e vector for top boundary - exactly matching Fortran."""
    # e(1) = ra*rb - 1
    rarb_r, rarb_i = cmul(ra_r, ra_i, rb_r, rb_i)
    e[0] = complex(rarb_r - 1.0, rarb_i)

    # e(2) = mu2*rb*(1-r1)
    t1_r = 1.0 - r1_r
    t1_i = -r1_i
    t2_r, t2_i = cmul(rb_r, rb_i, t1_r, t1_i)
    e[1] = complex(mu2 * t2_r, mu2 * t2_i)

    # e(3) = mu2*(r1 - ra*rb)
    e[2] = complex(mu2 * (r1_r - rarb_r), mu2 * (r1_i - rarb_i))

    # e(4) = mu2*ra*(r1-1)
    t1_r = r1_r - 1.0
    t1_i = r1_i
    t2_r, t2_i = cmul(ra_r, ra_i, t1_r, t1_i)
    e[3] = complex(mu2 * t2_r, mu2 * t2_i)

    # e(5) = mu2*mu2*(ra*rb - r1*r1)
    r1r1_r, r1r1_i = cmul(r1_r, r1_i, r1_r, r1_i)
    mu4 = mu2 * mu2
    e[4] = complex(mu4 * (rarb_r - r1r1_r), mu4 * (rarb_i - r1r1_i))

    # e(6) = -1
    e[5] = complex(-1.0, 0.0)

    # e(7) = mu2*rb/2
    e[6] = complex(mu2 * rb_r / 2.0, mu2 * rb_i / 2.0)


@jit(nopython=True, cache=True)
def propagate_g(c, g, temp):
    """Propagate g vector: g = g @ c"""
    # P-SV part
    for i in range(5):
        temp[i] = 0j
        for j in range(5):
            temp[i] += g[j] * c[j, i]
    for i in range(5):
        g[i] = temp[i]

    # SH part
    temp[5] = g[5] * c[5, 5] + g[6] * c[6, 5]
    temp[6] = g[5] * c[5, 6] + g[6] * c[6, 6]
    g[5] = temp[5]
    g[6] = temp[6]


@jit(nopython=True, cache=True)
def propagate_b(c, b, temp):
    """Propagate B matrix: b = b @ c"""
    # P-SV part
    for i in range(5):
        for j in range(5):
            temp[j] = 0j
            for l in range(5):
                temp[j] += b[i, l] * c[l, j]
        for j in range(5):
            b[i, j] = temp[j]

    # SH part
    for i in range(5, 7):
        temp[5] = b[i, 5] * c[5, 5] + b[i, 6] * c[6, 5]
        temp[6] = b[i, 5] * c[5, 6] + b[i, 6] * c[6, 6]
        b[i, 5] = temp[5]
        b[i, 6] = temp[6]


@jit(nopython=True, cache=True)
def propagate_z(a, z, temp):
    """Propagate z vector: z = z @ a"""
    for n in range(3):
        for j in range(4):
            temp[j] = 0j
            for l in range(4):
                temp[j] += z[n, l] * a[l, j]
        for j in range(4):
            z[n, j] = temp[j]
        # SH
        z[n, 4] = z[n, 4] * a[4, 4]


@jit(nopython=True, cache=True)
def initial_z(si, g, z):
    """Initialize z vector at source - exactly matching Fortran."""
    for i in range(3):
        # P-SV
        z[i, 0] = -si[i, 1] * g[0] - si[i, 2] * g[1] + si[i, 3] * g[2]
        z[i, 1] = si[i, 0] * g[0] - si[i, 2] * g[2] - si[i, 3] * g[3]
        z[i, 2] = si[i, 0] * g[1] + si[i, 1] * g[2] - si[i, 3] * g[4]
        z[i, 3] = -si[i, 0] * g[2] + si[i, 1] * g[3] + si[i, 2] * g[4]
        # SH
        z[i, 4] = si[i, 4] * g[5] + si[i, 5] * g[6]


@jit(nopython=True, cache=True)
def compute_source_coef(stype, xi, mu, flip, si):
    """Compute source coefficients - exactly matching Fortran."""
    si[:, :] = 0.0

    if stype == 2:  # Double couple
        si[0, 1] = 2.0 * xi / mu
        si[0, 3] = 4.0 * xi - 3.0
        si[1, 0] = flip / mu
        si[1, 4] = -si[1, 0]
        si[2, 3] = 1.0
        si[2, 5] = -1.0
    elif stype == 0:  # Explosion
        si[0, 1] = xi / mu
        si[0, 3] = 2.0 * xi
    elif stype == 1:  # Single force
        si[0, 2] = -flip
        si[1, 3] = -1.0
        si[1, 5] = 1.0


@jit(nopython=True, cache=True)
def compute_kernel_numba(
    k,
    omega_r,
    omega_i,
    d,
    vp,
    vs,
    rho,
    qp,
    qs,
    mu_arr,
    n_layers,
    src_layer,
    rcv_layer,
    stype,
    updn,
    flip,
    si,
    # Pre-allocated work arrays
    g,
    e,
    b,
    c,
    a,
    z,
    temp7,
    temp5,
):
    """
    Compute displacement kernels U(k, omega).
    Returns u[3,3] for modes n=0,1,2 and components Z,R,T.
    """
    u = np.zeros((3, 3), dtype=np.complex128)

    # Initialize B as identity
    b[:, :] = 0j
    for i in range(7):
        b[i, i] = 1.0 + 0j

    # Default boundary conditions
    e[:] = 0j
    e[0] = 1.0 + 0j
    e[5] = 1.0 + 0j

    g[:] = 0j
    g[4] = 1.0 + 0j
    g[6] = 1.0 + 0j

    z[:, :] = 0j

    mb = n_layers

    # Propagation from bottom to top
    for j in range(mb - 1, -1, -1):
        # Get layer parameters
        params = compute_layer_params(
            k, omega_r, omega_i, vp[j], vs[j], mu_arr[j], qp[j], qs[j], d[j]
        )
        ra_r, ra_i, rb_r, rb_i, r_r, r_i, r1_r, r1_i, kd, mu2 = params

        j1 = j + 1  # 1-based index

        if j1 == mb and d[j] < EPSILON:
            # Bottom halfspace
            initial_g(ra_r, ra_i, rb_r, rb_i, r_r, r_i, r1_r, r1_i, mu2, g)
        elif j1 == 1 and d[0] < EPSILON:
            # Top halfspace
            e_vector(ra_r, ra_i, rb_r, rb_i, r_r, r_i, r1_r, r1_i, mu2, e)
            break
        else:
            # Compute compound matrix and propagate
            compute_compound_matrix(
                ra_r, ra_i, rb_r, rb_i, r_r, r_i, r1_r, r1_i, kd, mu2, c
            )
            propagate_g(c, g, temp7)

        # Source layer
        if j1 == src_layer:
            initial_z(si, g, z)

        # Above source
        if j1 < src_layer:
            if j1 >= rcv_layer:
                compute_haskell_matrix(
                    ra_r, ra_i, rb_r, rb_i, r_r, r_i, r1_r, r1_i, kd, mu2, a
                )
                propagate_z(a, z, temp5)
            else:
                compute_compound_matrix(
                    ra_r, ra_i, rb_r, rb_i, r_r, r_i, r1_r, r1_i, kd, mu2, c
                )
                propagate_b(c, b, temp7)

    # Top boundary condition
    e[2] = 2.0 * e[2]

    rayl = g[0] * e[0] + g[1] * e[1] + g[2] * e[2] + g[3] * e[3] + g[4] * e[4]
    love = g[5] * e[5] + g[6] * e[6]

    # Apply B matrix to e
    for i in range(4):
        temp7[i] = 0j
        for jj in range(5):
            temp7[i] += b[i, jj] * e[jj]
    temp7[2] = temp7[2] / 2.0
    temp7[5] = b[5, 5] * e[5] + b[5, 6] * e[6]

    # Final computation
    for i in range(3):
        dum = z[i, 1] * temp7[0] + z[i, 2] * temp7[1] - z[i, 3] * temp7[2]
        z[i, 1] = -z[i, 0] * temp7[0] + z[i, 2] * temp7[2] + z[i, 3] * temp7[3]
        z[i, 0] = dum
        z[i, 4] = z[i, 4] * temp7[5]

    # Displacement kernels
    dum = k + 0j
    if stype == 1:
        dum = 1.0 + 0j

    for i in range(3):
        if abs(rayl) > EPSILON:
            u[i, 0] = dum * z[i, 1] / rayl  # Z
            u[i, 1] = dum * z[i, 0] / rayl  # R
        if abs(love) > EPSILON:
            u[i, 2] = dum * z[i, 4] / love  # T

    return u


def integrate_wavenumber_numba(
    omega_r,
    omega_i,
    d,
    vp,
    vs,
    rho,
    qp,
    qs,
    mu_arr,
    n_layers,
    src_layer,
    rcv_layer,
    stype,
    updn,
    flip,
    si,
    distances_km,
    k_min,
    k_max,
    dk,
):
    """
    Integrate over wavenumber using Numba-optimized kernel.
    """
    n_dist = len(distances_km)
    sums = np.zeros((n_dist, 9), dtype=np.complex128)

    # Pre-allocate work arrays
    g = np.zeros(7, dtype=np.complex128)
    e = np.zeros(7, dtype=np.complex128)
    b = np.zeros((7, 7), dtype=np.complex128)
    c = np.zeros((7, 7), dtype=np.complex128)
    a = np.zeros((5, 5), dtype=np.complex128)
    z = np.zeros((3, 5), dtype=np.complex128)
    temp7 = np.zeros(7, dtype=np.complex128)
    temp5 = np.zeros(5, dtype=np.complex128)

    k = k_min + 0.5 * dk
    n_k = int((k_max - k) / dk)

    for i_k in range(n_k):
        # Compute kernel
        u = compute_kernel_numba(
            k,
            omega_r,
            omega_i,
            d,
            vp,
            vs,
            rho,
            qp,
            qs,
            mu_arr,
            n_layers,
            src_layer,
            rcv_layer,
            stype,
            updn,
            flip,
            si,
            g,
            e,
            b,
            c,
            a,
            z,
            temp7,
            temp5,
        )

        # Sum with Bessel function weighting
        for ix in range(n_dist):
            x = distances_km[ix]
            z_arg = k * x

            if z_arg < EPSILON:
                aj0, aj1, aj2 = 1.0, 0.0, 0.0
            else:
                aj0, aj1, aj2 = j0(z_arg), j1(z_arg), jn(2, z_arg)

            # n=0 mode
            sums[ix, 0] += u[0, 0] * aj0 * flip
            sums[ix, 1] -= u[0, 1] * aj1
            sums[ix, 2] -= u[0, 2] * aj1

            # n=1 mode
            if z_arg > EPSILON:
                nf = (u[1, 1] + u[1, 2]) * aj1 / z_arg
            else:
                nf = 0j
            sums[ix, 3] += u[1, 0] * aj1 * flip
            sums[ix, 4] += u[1, 1] * aj0 - nf
            sums[ix, 5] += u[1, 2] * aj0 - nf

            # n=2 mode
            if z_arg > EPSILON:
                nf = 2.0 * (u[2, 1] + u[2, 2]) * aj2 / z_arg
            else:
                nf = 0j
            sums[ix, 6] += u[2, 0] * aj2 * flip
            sums[ix, 7] += u[2, 1] * aj1 - nf
            sums[ix, 8] += u[2, 2] * aj1 - nf

        k += dk

    return sums
