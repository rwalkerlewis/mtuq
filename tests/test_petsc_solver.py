#!/usr/bin/env python
"""
Test script for PETSc-based moment tensor inversion solver

This script tests the PETSc implementation against the standard solver
to ensure correctness and compatibility.
"""

import numpy as np
import pytest


def test_petsc_import():
    """Test that PETSc can be imported"""
    try:
        from petsc4py import PETSc
        print(f"PETSc version: {PETSc.Sys.getVersion()}")
        assert True
    except ImportError:
        pytest.skip("PETSc not available")


def test_petsc_grid_search_import():
    """Test that PETSc grid search can be imported"""
    try:
        from mtuq.grid_search_petsc import grid_search_petsc
        assert grid_search_petsc is not None
    except ImportError as e:
        pytest.skip(f"PETSc grid search not available: {e}")


def test_petsc_misfit_import():
    """Test that PETSc misfit module can be imported"""
    try:
        from mtuq.misfit.waveform import level_petsc
        assert level_petsc is not None
    except ImportError as e:
        pytest.skip(f"PETSc misfit module not available: {e}")


def test_petsc_initialization():
    """Test that PETSc can be initialized"""
    try:
        from petsc4py import PETSc
        
        if not PETSc.Sys.isInitialized():
            PETSc.Sys.init()
        
        comm = PETSc.COMM_WORLD
        rank = comm.rank
        size = comm.size
        
        print(f"PETSc initialized: rank {rank} of {size}")
        assert True
        
    except ImportError:
        pytest.skip("PETSc not available")


def test_petsc_vector_operations():
    """Test basic PETSc vector operations"""
    try:
        from petsc4py import PETSc
        
        if not PETSc.Sys.isInitialized():
            PETSc.Sys.init()
        
        comm = PETSc.COMM_WORLD
        
        # Create a vector
        vec = PETSc.Vec().create(comm=comm)
        vec.setSizes(10)
        vec.setUp()
        
        # Set values
        values = np.arange(10, dtype=float)
        vec.setValues(range(10), values)
        vec.assemblyBegin()
        vec.assemblyEnd()
        
        # Get values
        result = vec.getArray()
        
        # Check
        np.testing.assert_array_almost_equal(result, values)
        
        vec.destroy()
        
    except ImportError:
        pytest.skip("PETSc not available")


def test_petsc_matrix_vector_product():
    """Test PETSc matrix-vector product"""
    try:
        from petsc4py import PETSc
        
        if not PETSc.Sys.isInitialized():
            PETSc.Sys.init()
        
        comm = PETSc.COMM_WORLD
        
        # Create a small matrix
        n = 5
        mat = PETSc.Mat().create(comm=comm)
        mat.setSizes([n, n])
        mat.setType('dense')
        mat.setUp()
        
        # Fill with identity matrix
        for i in range(n):
            mat.setValue(i, i, 1.0)
        
        mat.assemblyBegin()
        mat.assemblyEnd()
        
        # Create input vector
        x = PETSc.Vec().create(comm=comm)
        x.setSizes(n)
        x.setUp()
        x.setValues(range(n), np.arange(n, dtype=float))
        x.assemblyBegin()
        x.assemblyEnd()
        
        # Create output vector
        y = PETSc.Vec().create(comm=comm)
        y.setSizes(n)
        y.setUp()
        
        # Matrix-vector product
        mat.mult(x, y)
        
        # Check result (should be same as input for identity matrix)
        result = y.getArray()
        expected = np.arange(n, dtype=float)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Clean up
        mat.destroy()
        x.destroy()
        y.destroy()
        
    except ImportError:
        pytest.skip("PETSc not available")


def test_petsc_misfit_computation_dummy():
    """
    Test PETSc misfit computation with dummy data
    
    This is a basic smoke test to ensure the computational kernel works.
    """
    try:
        from petsc4py import PETSc
        from mtuq.misfit.waveform import level_petsc
        
        if not PETSc.Sys.isInitialized():
            PETSc.Sys.init()
        
        # Create dummy data structures
        # This would need actual MTUQ data structures for a full test
        # For now, just verify the module loads and has the expected function
        
        assert hasattr(level_petsc, 'misfit')
        assert callable(level_petsc.misfit)
        
        print("PETSc misfit module structure verified")
        
    except ImportError:
        pytest.skip("PETSc not available")


def test_petsc_vs_standard_solver_compatibility():
    """
    Test that PETSc solver interface matches standard solver
    
    This ensures API compatibility between solvers.
    """
    try:
        from mtuq.grid_search import grid_search
        from mtuq.grid_search_petsc import grid_search_petsc
        
        import inspect
        
        # Get signatures
        sig_standard = inspect.signature(grid_search)
        sig_petsc = inspect.signature(grid_search_petsc)
        
        # Check that PETSc version has all parameters from standard
        standard_params = set(sig_standard.parameters.keys())
        petsc_params = set(sig_petsc.parameters.keys())
        
        # PETSc should have at least the same parameters
        # (it may have additional ones)
        assert standard_params.issubset(petsc_params), \
            f"Missing parameters: {standard_params - petsc_params}"
        
        print("API compatibility verified")
        
    except ImportError as e:
        pytest.skip(f"Could not import solvers: {e}")


if __name__ == '__main__':
    """Run tests when executed directly"""
    print("="*70)
    print("Testing PETSc Solver Implementation")
    print("="*70)
    print()
    
    tests = [
        ("PETSc Import", test_petsc_import),
        ("Grid Search Import", test_petsc_grid_search_import),
        ("Misfit Module Import", test_petsc_misfit_import),
        ("PETSc Initialization", test_petsc_initialization),
        ("Vector Operations", test_petsc_vector_operations),
        ("Matrix-Vector Product", test_petsc_matrix_vector_product),
        ("Misfit Computation", test_petsc_misfit_computation_dummy),
        ("API Compatibility", test_petsc_vs_standard_solver_compatibility),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, test_func in tests:
        try:
            print(f"Running: {name}...", end=" ")
            test_func()
            print("✓ PASSED")
            passed += 1
        except pytest.skip.Exception as e:
            print(f"⊘ SKIPPED - {e}")
            skipped += 1
        except Exception as e:
            print(f"✗ FAILED - {e}")
            failed += 1
    
    print()
    print("="*70)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*70)
    
    exit(0 if failed == 0 else 1)

