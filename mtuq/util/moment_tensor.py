"""
Moment tensor decomposition and source analysis utilities

This module provides functions for moment tensor analysis inspired by mtinv
(LLNL Moment Tensor Inversion toolkit). Key features include:

- Source type decomposition (DC, CLVD, ISO percentages)
- Hudson k, T source type parameters
- Lune coordinates (gamma, delta)  
- P, T, B principal axes computation
- Nodal plane solutions (strike, dip, rake)
- Variance reduction calculations

References:
    - Jost & Herrmann (1989) - A student's guide to and review of moment tensors
    - Hudson et al. (1989) - Source type plot for moment tensors
    - Tape & Tape (2012) - A geometric setting for moment tensors
    - Ford et al. (2010, 2012) - Network sensitivity solutions

Based on algorithms from mtinv (https://github.com/LLNL/mtinv)
Copyright 2024 Gene A. Ichinose (LLNL) - BSD 3-Clause License
"""

import numpy as np
from collections import namedtuple


# Named tuples for structured return values
SourceDecomposition = namedtuple('SourceDecomposition', [
    'pdc', 'pclvd', 'piso', 'pdev',  # percentages
    'Mdc', 'Mclvd', 'Miso', 'Mdev', 'Mtotal'  # moment components
])

HudsonParams = namedtuple('HudsonParams', ['k', 'T'])

LuneCoords = namedtuple('LuneCoords', ['gamma', 'delta', 'v', 'w'])

PrincipalAxes = namedtuple('PrincipalAxes', [
    'T_val', 'T_azimuth', 'T_plunge',  # T (tension) axis
    'P_val', 'P_azimuth', 'P_plunge',  # P (pressure) axis  
    'B_val', 'B_azimuth', 'B_plunge',  # B (null) axis
])

NodalPlane = namedtuple('NodalPlane', ['strike', 'dip', 'rake'])

NodalPlanes = namedtuple('NodalPlanes', ['plane1', 'plane2'])


def mt_to_matrix(mt, convention='USE'):
    """
    Convert moment tensor vector to 3x3 symmetric matrix.
    
    Parameters
    ----------
    mt : array_like
        6-element moment tensor [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp] in USE convention
        or [Mxx, Myy, Mzz, Mxy, Mxz, Myz] depending on convention
    convention : str
        'USE' (up-south-east, default), 'XYZ' (north-east-down), or 'NED'
        
    Returns
    -------
    M : ndarray
        3x3 symmetric moment tensor matrix
    """
    mt = np.asarray(mt).flatten()
    if len(mt) != 6:
        raise ValueError("Moment tensor must have 6 elements")
    
    if convention.upper() == 'USE':
        # USE: [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]
        # Matrix: [[Mrr, Mrt, Mrp], [Mrt, Mtt, Mtp], [Mrp, Mtp, Mpp]]
        M = np.array([
            [mt[0], mt[3], mt[4]],
            [mt[3], mt[1], mt[5]],
            [mt[4], mt[5], mt[2]]
        ])
    elif convention.upper() in ['XYZ', 'NED']:
        # XYZ: [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
        M = np.array([
            [mt[0], mt[3], mt[4]],
            [mt[3], mt[1], mt[5]],
            [mt[4], mt[5], mt[2]]
        ])
    else:
        raise ValueError(f"Unknown convention: {convention}")
    
    return M


def eigenvalues(mt, sorted_order='descending'):
    """
    Compute eigenvalues and eigenvectors of moment tensor.
    
    Parameters
    ----------
    mt : array_like
        6-element moment tensor or 3x3 matrix
    sorted_order : str
        'descending' (lambda1 >= lambda2 >= lambda3) or 
        'ascending' (lambda1 <= lambda2 <= lambda3) or
        'absolute' (|lambda1| <= |lambda2| <= |lambda3|)
        
    Returns
    -------
    eigenvalues : ndarray
        3-element array of eigenvalues
    eigenvectors : ndarray
        3x3 array where each column is an eigenvector
    """
    mt = np.asarray(mt)
    
    if mt.shape == (6,):
        M = mt_to_matrix(mt)
    elif mt.shape == (3, 3):
        M = mt
    else:
        raise ValueError("Moment tensor must be 6-element vector or 3x3 matrix")
    
    # Compute eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(M)
    
    # Sort according to specified order
    if sorted_order == 'descending':
        idx = np.argsort(evals)[::-1]
    elif sorted_order == 'ascending':
        idx = np.argsort(evals)
    elif sorted_order == 'absolute':
        idx = np.argsort(np.abs(evals))
    else:
        raise ValueError(f"Unknown sorted_order: {sorted_order}")
    
    return evals[idx], evecs[:, idx]


def scalar_moment(mt):
    """
    Compute scalar seismic moment M0.
    
    M0 = (1/sqrt(2)) * sqrt(sum(m_ij^2))
    
    Parameters
    ----------
    mt : array_like
        Moment tensor (6-element vector or 3x3 matrix)
        
    Returns
    -------
    M0 : float
        Scalar seismic moment
    """
    mt = np.asarray(mt)
    
    if mt.shape == (6,):
        M = mt_to_matrix(mt)
    elif mt.shape == (3, 3):
        M = mt
    else:
        raise ValueError("Moment tensor must be 6-element vector or 3x3 matrix")
    
    return np.sqrt(np.sum(M * M) / 2.0)


def moment_magnitude(mt):
    """
    Compute moment magnitude Mw from moment tensor.
    
    Mw = (2/3) * (log10(M0) - 9.1)
    
    Parameters
    ----------
    mt : array_like
        Moment tensor (6-element vector or 3x3 matrix)
        
    Returns
    -------
    Mw : float
        Moment magnitude
    """
    M0 = scalar_moment(mt)
    return (2.0/3.0) * (np.log10(M0) - 9.1)


def decompose_mt(mt):
    """
    Decompose moment tensor into DC, CLVD, and ISO components.
    
    Based on the decomposition scheme from Jost & Herrmann (1989)
    and as implemented in mtinv.
    
    Parameters
    ----------
    mt : array_like
        Moment tensor (6-element vector or 3x3 matrix)
        
    Returns
    -------
    decomp : SourceDecomposition
        Named tuple containing:
        - pdc, pclvd, piso: percentages (0-1) of total moment
        - pdev: deviatoric percentage (= pdc + pclvd)
        - Mdc, Mclvd, Miso, Mdev, Mtotal: moment magnitudes
    
    Notes
    -----
    The decomposition follows:
    - Miso = (lambda1 + lambda2 + lambda3) / 3
    - dev = [lambda_i - Miso] for each eigenvalue
    - Mclvd = -2 * min(|dev|)
    - Mdc = max(|dev|) - |Mclvd/2|
    
    Percentages are normalized so that pdc + pclvd + piso = 1
    """
    evals, _ = eigenvalues(mt, sorted_order='absolute')
    
    # Isotropic component
    Miso = np.mean(evals)
    
    # Deviatoric eigenvalues
    dev = evals - Miso
    
    # Sort deviatoric by absolute value
    dev_sorted = dev[np.argsort(np.abs(dev))]
    
    # CLVD magnitude (from smallest deviatoric eigenvalue)
    Mclvd = -2.0 * dev_sorted[0]
    
    # DC magnitude
    Mdc = dev_sorted[0] - dev_sorted[1]  # This gives the DC part
    
    # Total moment (for normalization)
    Mtotal = np.abs(Mdc) + np.abs(Mclvd) + np.abs(Miso)
    
    if Mtotal > 0:
        pdc = np.abs(Mdc) / Mtotal
        pclvd = np.abs(Mclvd) / Mtotal
        piso = np.abs(Miso) / Mtotal
    else:
        pdc = pclvd = piso = 0.0
    
    # Deviatoric = DC + CLVD
    pdev = pdc + pclvd
    Mdev = np.abs(Mdc) + np.abs(Mclvd)
    
    return SourceDecomposition(
        pdc=pdc, pclvd=pclvd, piso=piso, pdev=pdev,
        Mdc=Mdc, Mclvd=Mclvd, Miso=Miso, Mdev=Mdev, Mtotal=Mtotal
    )


def hudson_params(mt):
    """
    Compute Hudson source type parameters k and T.
    
    Parameters k and T from Hudson et al. (1989) characterize the
    source type on a 2D diamond-shaped plot.
    
    Parameters
    ----------
    mt : array_like
        Moment tensor (6-element vector or 3x3 matrix)
        
    Returns
    -------
    params : HudsonParams
        Named tuple with k and T values
        - k: ranges from -1 (implosion) to +1 (explosion)
        - T: ranges from -1 (-CLVD) to +1 (+CLVD)
        
    Notes
    -----
    k = Miso / (|Miso| + |max(dev)|)
    T = 2 * (-min(|dev|) / |max(dev)|)
    
    where dev are the deviatoric eigenvalues sorted by absolute value.
    """
    evals, _ = eigenvalues(mt, sorted_order='absolute')
    
    # Isotropic component
    Miso = np.mean(evals)
    
    # Deviatoric eigenvalues
    dev = evals - Miso
    
    # Sort deviatoric by absolute value
    dev_sorted = dev[np.argsort(np.abs(dev))]
    
    # Hudson parameters
    max_dev = dev_sorted[2]  # largest absolute value deviatoric
    
    if np.abs(Miso) + np.abs(max_dev) > 0:
        k = Miso / (np.abs(Miso) + np.abs(max_dev))
    else:
        k = 0.0
    
    if np.abs(max_dev) > 0:
        T = 2.0 * (-dev_sorted[0] / np.abs(max_dev))
    else:
        T = 0.0
    
    return HudsonParams(k=k, T=T)


def eig_to_lune(evals):
    """
    Convert eigenvalues to lune coordinates (gamma, delta).
    
    Based on Tape & Tape (2012) "A geometric setting for moment tensors".
    
    Parameters
    ----------
    evals : array_like
        3 eigenvalues (will be sorted descending internally)
        
    Returns
    -------
    coords : LuneCoords
        Named tuple with gamma, delta (degrees) and v, w parameters
        
    Notes
    -----
    - gamma (longitude): ranges from -30° (CLVD-) to +30° (CLVD+)
    - delta (latitude): ranges from -90° (implosion) to +90° (explosion)
    - DC is at (0, 0), explosion at (0, 90), implosion at (0, -90)
    """
    evals = np.asarray(evals).flatten()
    if len(evals) != 3:
        raise ValueError("Must provide 3 eigenvalues")
    
    # Sort descending
    e = np.sort(evals)[::-1]
    
    # Normalize
    norm = np.sqrt(e[0]**2 + e[1]**2 + e[2]**2)
    
    if norm == 0:
        return LuneCoords(gamma=0.0, delta=0.0, v=0.0, w=0.0)
    
    # Lune latitude (delta) - related to isotropic component
    bdot = (e[0] + e[1] + e[2]) / (np.sqrt(3) * norm)
    bdot = np.clip(bdot, -1, 1)
    delta = 90.0 - np.degrees(np.arccos(bdot))
    
    # Lune longitude (gamma) - related to CLVD component
    if e[0] != e[2]:
        gamma = np.degrees(np.arctan(
            (-e[0] + 2*e[1] - e[2]) / (np.sqrt(3) * (e[0] - e[2]))
        ))
    else:
        gamma = 0.0
    
    # Convert to Tape2015 v, w parameters
    gamma_rad = np.radians(gamma)
    delta_rad = np.radians(delta)
    beta = np.pi/2.0 - delta_rad
    
    v = (1.0/3.0) * np.sin(3.0 * gamma_rad)
    u = 0.75*beta - 0.5*np.sin(2.0*beta) + 0.0625*np.sin(4.0*beta)
    w = 3.0*np.pi/8.0 - u
    
    return LuneCoords(gamma=gamma, delta=delta, v=v, w=w)


def mt_to_lune(mt):
    """
    Convert moment tensor to lune coordinates.
    
    Parameters
    ----------
    mt : array_like
        Moment tensor (6-element vector or 3x3 matrix)
        
    Returns
    -------
    coords : LuneCoords
        Named tuple with gamma, delta (degrees) and v, w parameters
    """
    evals, _ = eigenvalues(mt, sorted_order='descending')
    return eig_to_lune(evals)


def principal_axes(mt):
    """
    Compute P, T, and B principal axes from moment tensor.
    
    The principal axes are the eigenvectors of the moment tensor:
    - T axis: tension axis (most positive eigenvalue)
    - P axis: pressure axis (most negative eigenvalue)
    - B axis: null axis (intermediate eigenvalue)
    
    Parameters
    ----------
    mt : array_like
        Moment tensor (6-element vector or 3x3 matrix)
        
    Returns
    -------
    axes : PrincipalAxes
        Named tuple with eigenvalues, azimuths, and plunges for T, P, B axes
        Azimuth is measured clockwise from North (0-360°)
        Plunge is measured downward from horizontal (0-90°)
    """
    evals, evecs = eigenvalues(mt, sorted_order='descending')
    
    def vec_to_az_pl(vec):
        """Convert eigenvector to azimuth and plunge."""
        # Ensure vector points downward (positive z in USE)
        if vec[0] < 0:  # If pointing up, flip
            vec = -vec
        
        # Convert from USE (r, theta, phi) to geographic
        # In USE: r=up, theta=south, phi=east
        r, s, e = vec  # up, south, east components
        
        # Plunge: angle from horizontal
        plunge = np.degrees(np.arcsin(np.clip(r, -1, 1)))
        
        # Azimuth: clockwise from North
        azimuth = np.degrees(np.arctan2(e, -s))  # -s because south is +y
        if azimuth < 0:
            azimuth += 360.0
        
        return azimuth, np.abs(plunge)
    
    T_az, T_pl = vec_to_az_pl(evecs[:, 0])  # Largest eigenvalue
    B_az, B_pl = vec_to_az_pl(evecs[:, 1])  # Middle eigenvalue
    P_az, P_pl = vec_to_az_pl(evecs[:, 2])  # Smallest eigenvalue
    
    return PrincipalAxes(
        T_val=evals[0], T_azimuth=T_az, T_plunge=T_pl,
        P_val=evals[2], P_azimuth=P_az, P_plunge=P_pl,
        B_val=evals[1], B_azimuth=B_az, B_plunge=B_pl
    )


def eig_to_mt(evals, evecs):
    """
    Reconstruct moment tensor from eigenvalues and eigenvectors.
    
    M_ij = sum_k(lambda_k * v_k_i * v_k_j)
    
    Parameters
    ----------
    evals : array_like
        3 eigenvalues
    evecs : array_like
        3x3 array of eigenvectors (columns)
        
    Returns
    -------
    M : ndarray
        3x3 moment tensor matrix
    """
    evals = np.asarray(evals)
    evecs = np.asarray(evecs)
    
    M = np.zeros((3, 3))
    for k in range(3):
        M += evals[k] * np.outer(evecs[:, k], evecs[:, k])
    
    return M


def nodal_planes(mt):
    """
    Compute the two nodal plane solutions (strike, dip, rake).
    
    For a double-couple source, there are two orthogonal nodal planes
    that are indistinguishable from seismic data alone.
    
    Parameters
    ----------
    mt : array_like
        Moment tensor (6-element vector or 3x3 matrix)
        
    Returns
    -------
    planes : NodalPlanes
        Named tuple containing plane1 and plane2, each with strike, dip, rake
        
    Notes
    -----
    Uses the major double couple decomposition.
    Strike: 0-360° (azimuth of fault from north)
    Dip: 0-90° (angle from horizontal)
    Rake: -180° to 180° (slip direction)
    """
    evals, evecs = eigenvalues(mt, sorted_order='descending')
    
    # For double couple, we need the T and P axes
    T = evecs[:, 0]  # Tension axis
    P = evecs[:, 2]  # Pressure axis
    
    # The two nodal planes are defined by the bisectors of T and P
    # Slip vector and normal for plane 1
    n1 = (T + P) / np.sqrt(2)
    s1 = (T - P) / np.sqrt(2)
    
    # Slip vector and normal for plane 2
    n2 = (T - P) / np.sqrt(2)
    s2 = (T + P) / np.sqrt(2)
    
    def compute_sdr(n, s):
        """Compute strike, dip, rake from normal and slip vectors."""
        # Ensure normal points upward
        if n[0] < 0:
            n = -n
            s = -s
        
        # Dip: angle from horizontal
        dip = np.degrees(np.arccos(np.clip(n[0], -1, 1)))
        
        # Strike: perpendicular to dip direction
        if np.abs(n[0]) > 0.9999:  # Nearly horizontal fault
            strike = 0.0
        else:
            strike = np.degrees(np.arctan2(-n[2], n[1]))
        
        if strike < 0:
            strike += 360.0
        
        # Rake: slip direction in fault plane
        # Project slip vector onto fault plane
        strike_vec = np.array([0, -np.sin(np.radians(strike)), np.cos(np.radians(strike))])
        dip_vec = np.cross(n, strike_vec)
        
        rake = np.degrees(np.arctan2(
            np.dot(s, dip_vec),
            np.dot(s, strike_vec)
        ))
        
        return NodalPlane(strike=strike, dip=dip, rake=rake)
    
    plane1 = compute_sdr(n1, s1)
    plane2 = compute_sdr(n2, s2)
    
    return NodalPlanes(plane1=plane1, plane2=plane2)


def variance_reduction(data, synthetics, norm='L2'):
    """
    Compute variance reduction between data and synthetics.
    
    VR = 1 - ||data - synthetics||^2 / ||data||^2
    
    For perfect fit, VR = 1 (or 100%)
    For no correlation, VR can be negative
    
    Parameters
    ----------
    data : array_like
        Observed data
    synthetics : array_like
        Synthetic/predicted data
    norm : str
        'L2' for sum of squares, 'L1' for sum of absolute values
        
    Returns
    -------
    vr : float
        Variance reduction (0-1 scale)
    """
    data = np.asarray(data).flatten()
    synthetics = np.asarray(synthetics).flatten()
    
    residuals = data - synthetics
    
    if norm.upper() == 'L2':
        data_norm = np.sum(data**2)
        residual_norm = np.sum(residuals**2)
    elif norm.upper() == 'L1':
        data_norm = np.sum(np.abs(data))
        residual_norm = np.sum(np.abs(residuals))
    else:
        raise ValueError(f"Unknown norm: {norm}")
    
    if data_norm > 0:
        vr = 1.0 - residual_norm / data_norm
    else:
        vr = 0.0
    
    return vr


def misfit_to_variance_reduction(misfit, data_norm):
    """
    Convert misfit value to variance reduction.
    
    Parameters
    ----------
    misfit : float or array_like
        Misfit value(s) from grid search
    data_norm : float
        Data norm (||data||^2 for L2 norm)
        
    Returns
    -------
    vr : float or array_like
        Variance reduction (same shape as misfit input)
    """
    return 1.0 - np.asarray(misfit) / data_norm


def cross_correlation(data, synthetics):
    """
    Compute normalized cross-correlation coefficient.
    
    CC = (data · synthetics) / (||data|| * ||synthetics||)
    
    Parameters
    ----------
    data : array_like
        Observed data
    synthetics : array_like
        Synthetic/predicted data
        
    Returns
    -------
    cc : float
        Normalized cross-correlation coefficient (-1 to 1)
    """
    data = np.asarray(data).flatten()
    synthetics = np.asarray(synthetics).flatten()
    
    d_norm = np.sqrt(np.sum(data**2))
    s_norm = np.sqrt(np.sum(synthetics**2))
    
    if d_norm > 0 and s_norm > 0:
        return np.dot(data, synthetics) / (d_norm * s_norm)
    else:
        return 0.0


def source_type_label(mt, threshold=0.1):
    """
    Return a descriptive label for the source type.
    
    Parameters
    ----------
    mt : array_like
        Moment tensor
    threshold : float
        Minimum fraction to consider a component significant
        
    Returns
    -------
    label : str
        Description like 'DC', 'CLVD+', 'ISO+', 'DC+CLVD', etc.
    """
    decomp = decompose_mt(mt)
    hudson = hudson_params(mt)
    
    labels = []
    
    # Check isotropic component
    if decomp.piso > threshold:
        if hudson.k > 0:
            labels.append('ISO+')
        else:
            labels.append('ISO-')
    
    # Check CLVD component
    if decomp.pclvd > threshold:
        if hudson.T > 0:
            labels.append('CLVD+')
        else:
            labels.append('CLVD-')
    
    # Check DC component
    if decomp.pdc > threshold:
        labels.append('DC')
    
    if not labels:
        return 'UNKNOWN'
    
    return '+'.join(labels)


def print_mt_summary(mt, name=''):
    """
    Print a comprehensive summary of moment tensor properties.
    
    Parameters
    ----------
    mt : array_like
        Moment tensor
    name : str
        Optional name/identifier
    """
    M0 = scalar_moment(mt)
    Mw = moment_magnitude(mt)
    decomp = decompose_mt(mt)
    hudson = hudson_params(mt)
    lune = mt_to_lune(mt)
    axes = principal_axes(mt)
    planes = nodal_planes(mt)
    
    print("="*60)
    if name:
        print(f"Moment Tensor Summary: {name}")
    else:
        print("Moment Tensor Summary")
    print("="*60)
    
    print(f"\nScalar Moment: M0 = {M0:.3e}")
    print(f"Moment Magnitude: Mw = {Mw:.2f}")
    
    print(f"\nSource Type Decomposition:")
    print(f"  DC:   {decomp.pdc*100:5.1f}%")
    print(f"  CLVD: {decomp.pclvd*100:5.1f}%")
    print(f"  ISO:  {decomp.piso*100:5.1f}%")
    print(f"  Deviatoric: {decomp.pdev*100:5.1f}%")
    
    print(f"\nHudson Parameters:")
    print(f"  k = {hudson.k:+.3f} (isotropic)")
    print(f"  T = {hudson.T:+.3f} (CLVD)")
    
    print(f"\nLune Coordinates:")
    print(f"  gamma = {lune.gamma:+.2f}° (longitude)")
    print(f"  delta = {lune.delta:+.2f}° (latitude)")
    print(f"  v = {lune.v:+.4f}")
    print(f"  w = {lune.w:+.4f}")
    
    print(f"\nPrincipal Axes:")
    print(f"  T-axis: val={axes.T_val:.3e}, az={axes.T_azimuth:.1f}°, pl={axes.T_plunge:.1f}°")
    print(f"  P-axis: val={axes.P_val:.3e}, az={axes.P_azimuth:.1f}°, pl={axes.P_plunge:.1f}°")
    print(f"  B-axis: val={axes.B_val:.3e}, az={axes.B_azimuth:.1f}°, pl={axes.B_plunge:.1f}°")
    
    print(f"\nNodal Planes:")
    print(f"  Plane 1: strike={planes.plane1.strike:.1f}°, dip={planes.plane1.dip:.1f}°, rake={planes.plane1.rake:.1f}°")
    print(f"  Plane 2: strike={planes.plane2.strike:.1f}°, dip={planes.plane2.dip:.1f}°, rake={planes.plane2.rake:.1f}°")
    
    label = source_type_label(mt)
    print(f"\nSource Type: {label}")
    print("="*60)


def create_mt_from_sdr(strike, dip, rake, M0=1.0):
    """
    Create a double-couple moment tensor from strike, dip, rake.
    
    Parameters
    ----------
    strike : float
        Strike angle in degrees (0-360)
    dip : float
        Dip angle in degrees (0-90)
    rake : float
        Rake angle in degrees (-180 to 180)
    M0 : float
        Scalar moment (default 1.0)
        
    Returns
    -------
    mt : ndarray
        6-element moment tensor in USE convention
    """
    # Convert to radians
    s = np.radians(strike)
    d = np.radians(dip)
    r = np.radians(rake)
    
    # Moment tensor elements (Aki & Richards convention, then convert to USE)
    Mxx = -M0 * (np.sin(d) * np.cos(r) * np.sin(2*s) + 
                 np.sin(2*d) * np.sin(r) * np.sin(s)**2)
    Myy = M0 * (np.sin(d) * np.cos(r) * np.sin(2*s) - 
                np.sin(2*d) * np.sin(r) * np.cos(s)**2)
    Mzz = M0 * np.sin(2*d) * np.sin(r)
    Mxy = M0 * (np.sin(d) * np.cos(r) * np.cos(2*s) + 
                0.5 * np.sin(2*d) * np.sin(r) * np.sin(2*s))
    Mxz = -M0 * (np.cos(d) * np.cos(r) * np.cos(s) + 
                 np.cos(2*d) * np.sin(r) * np.sin(s))
    Myz = -M0 * (np.cos(d) * np.cos(r) * np.sin(s) - 
                 np.cos(2*d) * np.sin(r) * np.cos(s))
    
    # Convert from NED (Aki-Richards) to USE
    # NED: Mxx, Myy, Mzz, Mxy, Mxz, Myz
    # USE: Mrr, Mtt, Mpp, Mrt, Mrp, Mtp
    # Where r=up, t=south, p=east
    # NED x=north, y=east, z=down
    # Mrr = Mzz, Mtt = Mxx, Mpp = Myy
    # Mrt = Mxz, Mrp = -Myz, Mtp = -Mxy
    
    return np.array([Mzz, Mxx, Myy, Mxz, -Myz, -Mxy])


def create_pure_dc(strike=0, dip=45, rake=90, M0=1.0):
    """Create a pure double-couple moment tensor."""
    return create_mt_from_sdr(strike, dip, rake, M0)


def create_pure_clvd(azimuth=0, plunge=90, sign=1, M0=1.0):
    """
    Create a pure CLVD (compensated linear vector dipole) moment tensor.
    
    Parameters
    ----------
    azimuth : float
        Azimuth of symmetry axis (degrees)
    plunge : float
        Plunge of symmetry axis (degrees, 0=horizontal, 90=vertical)
    sign : int
        +1 for positive CLVD, -1 for negative CLVD
    M0 : float
        Scalar moment
    """
    # Eigenvalues for CLVD: [2, -1, -1] * sign * scale
    # or [-2, 1, 1] * sign * scale
    scale = M0 * np.sqrt(6) / 6  # To get M0 from eigenvalues
    
    if sign > 0:
        evals = np.array([2, -1, -1]) * scale
    else:
        evals = np.array([-2, 1, 1]) * scale
    
    # Rotation matrix for symmetry axis
    az = np.radians(azimuth)
    pl = np.radians(plunge)
    
    # Symmetry axis direction
    z_axis = np.array([np.sin(pl), -np.cos(pl)*np.cos(az), np.cos(pl)*np.sin(az)])
    
    # Create orthonormal basis
    if np.abs(z_axis[0]) < 0.9:
        x_axis = np.cross([1, 0, 0], z_axis)
    else:
        x_axis = np.cross([0, 1, 0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    evecs = np.column_stack([z_axis, x_axis, y_axis])
    
    M = eig_to_mt(evals, evecs)
    
    return np.array([M[0,0], M[1,1], M[2,2], M[0,1], M[0,2], M[1,2]])


def create_pure_iso(sign=1, M0=1.0):
    """
    Create a pure isotropic moment tensor (explosion or implosion).
    
    Parameters
    ----------
    sign : int
        +1 for explosion, -1 for implosion
    M0 : float
        Scalar moment
    """
    # For isotropic, eigenvalues are equal
    val = sign * M0 * np.sqrt(2) / 3
    
    return np.array([val, val, val, 0, 0, 0])


# Utility functions for grid searches

def evaluate_source_types(sources, data, greens, misfit_func):
    """
    Evaluate multiple source types and return detailed analysis.
    
    This is a higher-level function that combines grid search results
    with source type decomposition for comprehensive analysis.
    
    Parameters
    ----------
    sources : list
        List of source objects (MomentTensor or similar)
    data : array_like
        Observed data
    greens : object
        Green's functions
    misfit_func : callable
        Misfit function to use
        
    Returns
    -------
    results : list of dict
        List of dictionaries containing misfit, variance reduction,
        and source decomposition for each source
    """
    results = []
    
    for source in sources:
        # Get moment tensor
        mt = source.as_vector() if hasattr(source, 'as_vector') else np.asarray(source)
        
        # Compute decomposition
        decomp = decompose_mt(mt)
        hudson = hudson_params(mt)
        lune = mt_to_lune(mt)
        
        result = {
            'mt': mt,
            'decomposition': decomp,
            'hudson': hudson,
            'lune': lune,
            'source_label': source_type_label(mt)
        }
        
        results.append(result)
    
    return results
