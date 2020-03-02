
import numpy as np
from mtuq.event import MomentTensor
from mtuq.util.math import open_interval


deg2rad = np.pi/180.
rad2deg = 180./np.pi


def to_mt(rho, v, w, kappa, sigma, h):
    """ Converts from lune parameters to MomentTensor object
    """
    mt = to_mij(rho, v, w, kappa, sigma, h)
    return MomentTensor(mt, convention='USE')


def to_force(F0, theta, h):
    """ Converts from spherical coordinates to Force object
    """
    raise NotImplementedError


def to_mij(rho, v, w, kappa, sigma, h):
    """ Converts from lune parameters to moment tensor parameters 
    (up-south-east convention)
    """
    kR3 = np.sqrt(3.)
    k2R6 = 2.*np.sqrt(6.)
    k2R3 = 2.*np.sqrt(3.)
    k4R6 = 4.*np.sqrt(6.)
    k8R6 = 8.*np.sqrt(6.)

    m0 = rho/np.sqrt(2.)
    delta, gamma = to_delta_gamma(v, w)
    beta = 90. - delta
    theta = np.arccos(h)/deg2rad

    Cb  = np.cos(beta*deg2rad)
    Cg  = np.cos(gamma*deg2rad)
    Cs  = np.cos(sigma*deg2rad)
    Ct  = np.cos(theta*deg2rad)
    Ck  = np.cos(kappa*deg2rad)
    C2k = np.cos(2.0*kappa*deg2rad)
    C2s = np.cos(2.0*sigma*deg2rad)
    C2t = np.cos(2.0*theta*deg2rad)

    Sb  = np.sin(beta*deg2rad)
    Sg  = np.sin(gamma*deg2rad)
    Ss  = np.sin(sigma*deg2rad)
    St  = np.sin(theta*deg2rad)
    Sk  = np.sin(kappa*deg2rad)
    S2k = np.sin(2.0*kappa*deg2rad)
    S2s = np.sin(2.0*sigma*deg2rad)
    S2t = np.sin(2.0*theta*deg2rad)

    mt0 = m0 * (1./12.) * \
        (k4R6*Cb + Sb*(kR3*Sg*(-1. - 3.*C2t + 6.*C2s*St*St) + 12.*Cg*S2t*Ss))

    mt1 = m0* (1./24.) * \
        (k8R6*Cb + Sb*(-24.*Cg*(Cs*St*S2k + S2t*Sk*Sk*Ss) + kR3*Sg * \
        ((1. + 3.*C2k)*(1. - 3.*C2s) + 12.*C2t*Cs*Cs*Sk*Sk - 12.*Ct*S2k*S2s)))

    mt2 = m0* (1./6.) * \
        (k2R6*Cb + Sb*(kR3*Ct*Ct*Ck*Ck*(1. + 3.*C2s)*Sg - k2R3*Ck*Ck*Sg*St*St + 
        kR3*(1. - 3.*C2s)*Sg*Sk*Sk + 6.*Cg*Cs*St*S2k + 
        3.*Ct*(-4.*Cg*Ck*Ck*St*Ss + kR3*Sg*S2k*S2s)))

    mt3 = m0* (-1./2.)*Sb*(k2R3*Cs*Sg*St*(Ct*Cs*Sk - Ck*Ss) +
        2.*Cg*(Ct*Ck*Cs + C2t*Sk*Ss))

    mt4 = -m0* (1./2.)*Sb*(Ck*(kR3*Cs*Cs*Sg*S2t + 2.*Cg*C2t*Ss) +
        Sk*(-2.*Cg*Ct*Cs + kR3*Sg*St*S2s))

    mt5 = -m0* (1./8.)*Sb*(4.*Cg*(2.*C2k*Cs*St + S2t*S2k*Ss) +
        kR3*Sg*((1. - 2.*C2t*Cs*Cs - 3.*C2s)*S2k + 4.*Ct*C2k*S2s))

    if type(mt0) is np.ndarray:
        return np.column_stack([mt0, mt1, mt2, mt3, mt4, mt5])
    else:
        return np.array([mt0, mt1, mt2, mt3, mt4, mt5])


def spherical_to_Cartesian(r, theta, phi):
    """ Converts from spherical to Cartesian coordinates
    """
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    if type(r) is np.ndarray:
        return np.column_stack([x, y, z])
    else:
        return np.array([x, y, z])


def to_xyz(F0, theta, h):
    """ Converts from spherical to Cartesian coordinates
    """
    return spherical_to_Cartesian(F0, theta, np.arccos(h))


def to_rtp(F0, theta, h):
    """ Converts from spherical to Cartesian coordinates
    """
    x, y, z = to_xyz(F0, theta, h)
    if type(x) is np.ndarray:
        return np.column_stack([z, x, y,])
    else:
        return np.array([z, x, y])


def to_delta_gamma(v, w):
    """ Converts from Tape2015 parameters to lune coordinates
    """
    return to_delta(w), to_gamma(v)


def to_gamma(v):
    """ Converts from Tape2015 parameter v to lune longitude
    """
    gamma = (1./3.)*np.arcsin(3.*v)
    gamma *= rad2deg
    return gamma


def to_delta(w):
    """ Converts from Tape2015 parameter w to lune latitude
    """
    beta0 = np.linspace(0, np.pi, 100)
    u0 = 0.75*beta0 - 0.5*np.sin(2.*beta0) + 0.0625*np.sin(4.*beta0)
    beta = np.interp(3.*np.pi/8. - w, u0, beta0)
    beta *= rad2deg
    delta = 90. - beta
    return delta


def to_v_w(delta, gamma):
    """ Converts from lune coordinates to Tape2015 parameters
    """
    return to_v(gamma), to_w(delta)


def to_v(gamma):
    """ Converts from lune longitude to Tape2015 parameter v
    """
    gamma *= deg2rad
    v = (1./3.)*np.sin(3.*gamma)
    return v


def to_w(delta):
    """ Converts from lune latitude to Tape2015 parameter w
    """
    delta *= deg2rad
    beta = np.pi/2. - delta
    u = (0.75*beta - 0.5*np.sin(2.*beta) + 0.0625*np.sin(4.*beta))
    w = 3.*np.pi/8. - u
    return w


def to_M0(Mw):
    """ Converts from moment magnitude to scalar moment
    """
    return 10.**(1.5*float(Mw) + 9.1)


def to_rho(Mw):
    """ Converts from moment magnitude to Tape2012 magnitude parameter
    """
    return to_M0(Mw)*np.sqrt(2.)


def semiregular_grid(npts_v, npts_w, tightness=0.5):
    """ Semiregular moment tensor grid

    For tightness~0, grid will be regular in Tape2012 parameters delta, gamma.
    For tightness~1, grid will be regular in Tape2015 parameters v, w.
    For intermediate values, the grid will be "semiregular" in the sense of
    a linear interpolation between the above cases.

    Another way to think about is that as `tightness` increases, the extremal
    grid points get closer to the boundary of the lune.
    """
    assert 0. <= tightness < 1.,\
        Exception("Allowable range: 0. <= tightness < 1.")

    v1 = open_interval(-1./3., 1./3., npts_v)
    w1 = open_interval(-3./8.*np.pi, 3./8.*np.pi, npts_w)

    gamma1 = to_gamma(v1)
    delta1 = to_delta(w1)

    gamma2 = np.linspace(-30., 30., npts_v)
    delta2 = np.linspace(-87.5, 87.5, npts_w)

    delta = delta1*(1.-tightness) + delta2*tightness
    gamma = gamma1*(1.-tightness) + gamma2*tightness

    return to_v(gamma), to_w(delta)


def HammerGrid():
    """ Unstructured grid for visualization on the lune
    """
    raise NotImplementedError
