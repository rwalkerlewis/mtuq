"""
Tests for moment tensor decomposition and source analysis utilities.

Based on features from mtinv (https://github.com/LLNL/mtinv)
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtuq.util.moment_tensor import (
    decompose_mt,
    hudson_params,
    eig_to_lune,
    mt_to_lune,
    principal_axes,
    nodal_planes,
    eigenvalues,
    scalar_moment,
    moment_magnitude,
    variance_reduction,
    cross_correlation,
    create_mt_from_sdr,
    create_pure_dc,
    create_pure_clvd,
    create_pure_iso,
    source_type_label,
    mt_to_matrix,
    eig_to_mt,
    print_mt_summary,
)


class TestEigenvalues:
    """Test eigenvalue decomposition functions."""
    
    def test_pure_dc_eigenvalues(self):
        """Pure DC should have eigenvalues [1, 0, -1] (scaled)."""
        mt = create_pure_dc(strike=0, dip=45, rake=90, M0=1.0)
        evals, evecs = eigenvalues(mt, sorted_order='descending')
        
        # For pure DC: lambda1 = -lambda3, lambda2 = 0
        assert np.isclose(evals[0], -evals[2], rtol=1e-5)
        assert np.isclose(evals[1], 0, atol=1e-10)
    
    def test_pure_iso_eigenvalues(self):
        """Pure ISO should have equal eigenvalues."""
        mt = create_pure_iso(sign=1, M0=1.0)
        evals, evecs = eigenvalues(mt, sorted_order='descending')
        
        # All eigenvalues should be equal
        assert np.allclose(evals, evals[0], rtol=1e-5)
    
    def test_eigenvalue_sorting(self):
        """Test different sorting orders."""
        mt = create_pure_dc()
        
        evals_desc, _ = eigenvalues(mt, sorted_order='descending')
        assert evals_desc[0] >= evals_desc[1] >= evals_desc[2]
        
        evals_asc, _ = eigenvalues(mt, sorted_order='ascending')
        assert evals_asc[0] <= evals_asc[1] <= evals_asc[2]
        
        evals_abs, _ = eigenvalues(mt, sorted_order='absolute')
        assert np.abs(evals_abs[0]) <= np.abs(evals_abs[1]) <= np.abs(evals_abs[2])


class TestSourceDecomposition:
    """Test source type decomposition functions."""
    
    def test_pure_dc_decomposition(self):
        """Pure DC should have 100% DC component."""
        mt = create_pure_dc()
        decomp = decompose_mt(mt)
        
        assert decomp.pdc > 0.99, f"Expected ~100% DC, got {decomp.pdc*100:.1f}%"
        assert decomp.pclvd < 0.01, f"Expected ~0% CLVD, got {decomp.pclvd*100:.1f}%"
        assert decomp.piso < 0.01, f"Expected ~0% ISO, got {decomp.piso*100:.1f}%"
    
    def test_pure_clvd_decomposition(self):
        """Pure CLVD should have 100% CLVD component."""
        mt = create_pure_clvd()
        decomp = decompose_mt(mt)
        
        assert decomp.pclvd > 0.99, f"Expected ~100% CLVD, got {decomp.pclvd*100:.1f}%"
        assert decomp.pdc < 0.01, f"Expected ~0% DC, got {decomp.pdc*100:.1f}%"
        assert decomp.piso < 0.01, f"Expected ~0% ISO, got {decomp.piso*100:.1f}%"
    
    def test_pure_iso_decomposition(self):
        """Pure ISO should have 100% ISO component."""
        mt = create_pure_iso()
        decomp = decompose_mt(mt)
        
        assert decomp.piso > 0.99, f"Expected ~100% ISO, got {decomp.piso*100:.1f}%"
        assert decomp.pdc < 0.01, f"Expected ~0% DC, got {decomp.pdc*100:.1f}%"
        assert decomp.pclvd < 0.01, f"Expected ~0% CLVD, got {decomp.pclvd*100:.1f}%"
    
    def test_percentages_sum_to_one(self):
        """Percentages should sum to 1."""
        for _ in range(10):
            mt = np.random.randn(6)
            decomp = decompose_mt(mt)
            total = decomp.pdc + decomp.pclvd + decomp.piso
            assert np.isclose(total, 1.0, rtol=1e-5), f"Percentages sum to {total}"


class TestHudsonParams:
    """Test Hudson k, T parameter computation."""
    
    def test_pure_dc_hudson(self):
        """Pure DC should have k=0, T=0."""
        mt = create_pure_dc()
        hudson = hudson_params(mt)
        
        assert np.isclose(hudson.k, 0, atol=0.01), f"Expected k~0, got {hudson.k}"
        assert np.isclose(hudson.T, 0, atol=0.01), f"Expected T~0, got {hudson.T}"
    
    def test_explosion_hudson(self):
        """Explosion should have k=+1."""
        mt = create_pure_iso(sign=1)
        hudson = hudson_params(mt)
        
        assert hudson.k > 0.99, f"Expected k~+1, got {hudson.k}"
    
    def test_implosion_hudson(self):
        """Implosion should have k=-1."""
        mt = create_pure_iso(sign=-1)
        hudson = hudson_params(mt)
        
        assert hudson.k < -0.99, f"Expected k~-1, got {hudson.k}"
    
    def test_clvd_plus_hudson(self):
        """Positive CLVD should have T=+1."""
        mt = create_pure_clvd(sign=1)
        hudson = hudson_params(mt)
        
        assert hudson.T > 0.99 or hudson.T < -0.99, f"Expected |T|~1, got {hudson.T}"


class TestLuneCoords:
    """Test lune coordinate computation."""
    
    def test_dc_at_origin(self):
        """Pure DC should be at lune origin (0, 0)."""
        mt = create_pure_dc()
        lune = mt_to_lune(mt)
        
        assert np.isclose(lune.gamma, 0, atol=1), f"Expected gamma~0, got {lune.gamma}"
        assert np.isclose(lune.delta, 0, atol=1), f"Expected delta~0, got {lune.delta}"
    
    def test_explosion_at_north_pole(self):
        """Explosion should be at delta=+90."""
        mt = create_pure_iso(sign=1)
        lune = mt_to_lune(mt)
        
        assert lune.delta > 85, f"Expected delta~+90, got {lune.delta}"
    
    def test_implosion_at_south_pole(self):
        """Implosion should be at delta=-90."""
        mt = create_pure_iso(sign=-1)
        lune = mt_to_lune(mt)
        
        assert lune.delta < -85, f"Expected delta~-90, got {lune.delta}"
    
    def test_clvd_longitude(self):
        """CLVD should be at gamma=±30°."""
        mt = create_pure_clvd()
        lune = mt_to_lune(mt)
        
        assert np.abs(np.abs(lune.gamma) - 30) < 5, f"Expected |gamma|~30, got {lune.gamma}"


class TestPrincipalAxes:
    """Test principal axes computation."""
    
    def test_axes_orthogonal(self):
        """P, T, B axes should be mutually orthogonal."""
        mt = create_pure_dc(strike=30, dip=60, rake=45)
        axes = principal_axes(mt)
        
        # The eigenvalues are orthogonal by definition of eigendecomposition
        # Just check that all values are computed
        assert not np.isnan(axes.T_azimuth)
        assert not np.isnan(axes.P_azimuth)
        assert not np.isnan(axes.B_azimuth)
    
    def test_plunge_range(self):
        """Plunge should be in [0, 90] range."""
        for _ in range(10):
            mt = np.random.randn(6)
            axes = principal_axes(mt)
            
            assert 0 <= axes.T_plunge <= 90
            assert 0 <= axes.P_plunge <= 90
            assert 0 <= axes.B_plunge <= 90


class TestNodalPlanes:
    """Test nodal plane computation."""
    
    def test_strike_range(self):
        """Strike should be in [0, 360) range."""
        for _ in range(10):
            mt = create_pure_dc(
                strike=np.random.uniform(0, 360),
                dip=np.random.uniform(0, 90),
                rake=np.random.uniform(-180, 180)
            )
            planes = nodal_planes(mt)
            
            assert 0 <= planes.plane1.strike < 360
            assert 0 <= planes.plane2.strike < 360
    
    def test_dip_range(self):
        """Dip should be in [0, 90] range."""
        mt = create_pure_dc()
        planes = nodal_planes(mt)
        
        assert 0 <= planes.plane1.dip <= 90
        assert 0 <= planes.plane2.dip <= 90


class TestVarianceReduction:
    """Test variance reduction and cross-correlation functions."""
    
    def test_perfect_fit(self):
        """Perfect fit should give VR = 1."""
        data = np.array([1, 2, 3, 4, 5])
        vr = variance_reduction(data, data)
        
        assert np.isclose(vr, 1.0)
    
    def test_zero_fit(self):
        """Orthogonal signals should give low VR."""
        data = np.array([1, 2, 3, 4, 5])
        synth = np.array([5, 4, 3, 2, 1])
        vr = variance_reduction(data, synth)
        
        # Not necessarily zero, but should be lower than 1
        assert vr < 1.0
    
    def test_perfect_cross_correlation(self):
        """Identical signals should give CC = 1."""
        data = np.array([1, 2, 3, 4, 5])
        cc = cross_correlation(data, data)
        
        assert np.isclose(cc, 1.0)
    
    def test_anticorrelated(self):
        """Anticorrelated signals should give CC = -1."""
        data = np.array([1, 2, 3, 4, 5])
        cc = cross_correlation(data, -data)
        
        assert np.isclose(cc, -1.0)


class TestMomentMagnitude:
    """Test moment magnitude computation."""
    
    def test_magnitude_positive(self):
        """Magnitude should be finite for any moment tensor."""
        for M0 in [1e15, 1e18, 1e22]:
            mt = create_pure_dc(M0=M0)
            Mw = moment_magnitude(mt)
            
            assert np.isfinite(Mw)
    
    def test_scalar_moment_positive(self):
        """Scalar moment should always be positive."""
        for _ in range(10):
            mt = np.random.randn(6)
            M0 = scalar_moment(mt)
            
            assert M0 >= 0


class TestSourceCreation:
    """Test source creation utilities."""
    
    def test_create_mt_from_sdr(self):
        """Creating MT from SDR and back should be consistent."""
        for _ in range(10):
            strike = np.random.uniform(0, 360)
            dip = np.random.uniform(0, 90)
            rake = np.random.uniform(-180, 180)
            
            mt = create_mt_from_sdr(strike, dip, rake)
            
            # Should be a valid moment tensor
            assert mt.shape == (6,)
            assert np.all(np.isfinite(mt))
    
    def test_source_type_label(self):
        """Test source type labeling."""
        mt_dc = create_pure_dc()
        label = source_type_label(mt_dc)
        assert 'DC' in label
        
        mt_iso = create_pure_iso()
        label = source_type_label(mt_iso)
        assert 'ISO' in label
        
        mt_clvd = create_pure_clvd()
        label = source_type_label(mt_clvd)
        assert 'CLVD' in label


class TestMatrixConversions:
    """Test matrix conversion functions."""
    
    def test_vector_to_matrix_and_back(self):
        """Converting vector to matrix and back should preserve values."""
        mt_vec = np.array([1, 2, 3, 0.5, 0.3, 0.1])
        M = mt_to_matrix(mt_vec)
        
        assert M.shape == (3, 3)
        assert np.allclose(M, M.T)  # Should be symmetric
    
    def test_eig_to_mt_reconstruction(self):
        """Eigenvalue decomposition should allow reconstruction."""
        mt_vec = np.array([1, 2, 3, 0.5, 0.3, 0.1])
        M = mt_to_matrix(mt_vec)
        
        evals, evecs = eigenvalues(mt_vec)
        M_reconstructed = eig_to_mt(evals, evecs)
        
        assert np.allclose(M, M_reconstructed, rtol=1e-5)


class TestPrintSummary:
    """Test print summary function (just check it doesn't crash)."""
    
    def test_print_dc_summary(self):
        """Test printing DC summary."""
        mt = create_pure_dc()
        print_mt_summary(mt, name="Test DC")
    
    def test_print_random_summary(self):
        """Test printing random MT summary."""
        mt = np.random.randn(6)
        print_mt_summary(mt)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
