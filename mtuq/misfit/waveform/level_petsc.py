"""
Waveform misfit module (PETSc-accelerated version)

This module provides PETSc-accelerated waveform misfit computation using
distributed arrays and parallel linear algebra operations.

See ``mtuq/misfit/waveform/__init__.py`` for more information
"""

import numpy as np
import time

from mtuq.misfit.waveform._stats import _flatten, calculate_norm_data
from mtuq.util.math import to_mij, to_rtp
from mtuq.util.signal import get_components, get_time_sampling

try:
    from petsc4py import PETSc
    PETSC_AVAILABLE = True
except ImportError:
    PETSC_AVAILABLE = False
    PETSc = None


def misfit(data, greens, sources, norm, time_shift_groups,
    time_shift_min, time_shift_max, msg_handle, debug_level=0,
    normalize=False, use_petsc_solvers=True):
    """
    Waveform misfit function (PETSc-accelerated version)

    Uses PETSc vectors and matrices for distributed linear algebra operations
    in the misfit computation.

    Parameters
    ----------
    use_petsc_solvers : bool
        If True, use PETSc KSP solvers for linear algebra.
        If False, use NumPy operations wrapped in PETSc vectors.

    See ``mtuq/misfit/waveform/__init__.py`` for other parameter descriptions
    """
    
    if not PETSC_AVAILABLE:
        raise ImportError(
            "PETSc not available. Please install petsc4py:\n"
            "  conda install -c conda-forge petsc4py\n"
            "or\n"
            "  pip install petsc4py"
        )

    if not PETSc.Sys.isInitialized():
        PETSc.Sys.init()

    comm = PETSc.COMM_WORLD

    if normalize:
        components = _flatten(time_shift_groups)
        norm_data = calculate_norm_data(data, norm, components)

    #
    # collect metadata
    #
    nt, dt = _get_time_sampling(data)
    stations = _get_stations(data)
    components = _get_components(data)

    # collect user-supplied data weights
    weights = _get_weights(data, stations, components)

    # which components will be used to determine time shifts? (boolean array)
    groups = _get_groups(time_shift_groups, components)

    #
    # collapse main structures into NumPy arrays
    #
    data_array = _get_data(data, stations, components)
    greens_array = _get_greens(greens, stations, components)
    sources_array = _to_array(sources)

    # sanity checks
    _check(data_array, greens_array, sources_array)

    #
    # Convert to PETSc data structures
    #
    start_time = time.time()
    
    # Create PETSc vectors for data
    data_petsc = _numpy_to_petsc_vec(data_array.flatten(), comm)
    
    # Create PETSc vectors for sources (each source is a vector)
    nsources = sources_array.shape[0]
    nparams = sources_array.shape[1]
    
    if debug_level > 0:
        print(f'  Number of sources: {nsources}')
        print(f'  Number of parameters per source: {nparams}')
        print(f'  Data array shape: {data_array.shape}')
        print(f'  Greens array shape: {greens_array.shape}')
    
    #
    # cross-correlate data and synthetics using PETSc operations
    #
    padding = _get_padding(time_shift_min, time_shift_max, dt)
    
    # Use NumPy for correlations (these are embarassingly parallel anyway)
    data_data = _autocorr_1(data_array)
    greens_greens = _autocorr_2(greens_array, padding)
    greens_data = _corr_1_2(data_array, greens_array, padding)
    
    if debug_level > 0:
        print(f'  Time for PETSc setup (s): {time.time() - start_time:.4f}')

    #
    # Compute misfit using PETSc linear algebra
    #
    start_time = time.time()
    
    results = _compute_misfit_petsc(
        data_data, greens_data, greens_greens, sources_array, 
        groups, weights, norm, dt, padding, 
        use_petsc_solvers=use_petsc_solvers, 
        debug_level=debug_level, msg_handle=msg_handle)

    if debug_level > 0:
        print(f'  Time for PETSc misfit computation (s): {time.time() - start_time:.4f}')

    if normalize:
        results /= norm_data

    return results


def _compute_misfit_petsc(data_data, greens_data, greens_greens, sources, 
                          groups, weights, norm, dt, padding, 
                          use_petsc_solvers=True, debug_level=0, msg_handle=None):
    """
    Core misfit computation using PETSc linear algebra
    
    This function implements the L2 misfit computation using PETSc vectors
    and matrix operations for parallel efficiency.
    """
    
    if not PETSC_AVAILABLE:
        raise ImportError("PETSc not available")
    
    comm = PETSc.COMM_WORLD
    nsources = sources.shape[0]
    results = np.zeros((nsources, 1))
    
    nstations = data_data.shape[0]
    ncomponents = data_data.shape[1]
    ngreens = greens_greens.shape[3]
    npadding = padding[0] + padding[1] + 1
    
    # Create PETSc vectors for efficient computation
    source_vec = PETSc.Vec().create(comm=comm)
    source_vec.setSizes(ngreens)
    source_vec.setUp()
    
    # Temporary vectors for intermediate results
    temp_vec = PETSc.Vec().create(comm=comm)
    temp_vec.setSizes(ngreens)
    temp_vec.setUp()
    
    result_vec = PETSc.Vec().create(comm=comm)
    result_vec.setSizes(ngreens)
    result_vec.setUp()
    
    hybrid_norm = 1 if norm == 'hybrid' else 0
    
    # Process each source
    for isource in range(nsources):
        
        if msg_handle:
            msg_handle()
        
        # Set source parameters in PETSc vector
        source_vec.setValues(range(ngreens), sources[isource, :])
        source_vec.assemblyBegin()
        source_vec.assemblyEnd()
        
        misfit_value = 0.0
        
        # Iterate over stations
        for istation in range(nstations):
            
            # Process each time shift group
            for igroup in range(groups.shape[0]):
                
                # Find optimal time shift using cross-correlation
                cc_sum = np.zeros(npadding)
                for icomp in range(ncomponents):
                    if groups[igroup, icomp] > 0:
                        # Cross-correlation contribution
                        cc_sum += np.dot(sources[isource, :], 
                                        greens_data[istation, icomp, :, :])
                
                # Index of maximum correlation
                ishift = cc_sum.argmax()
                
                # Compute misfit for this group
                for icomp in range(ncomponents):
                    if groups[igroup, icomp] > 0:
                        
                        weight = weights[istation, icomp]
                        if weight == 0:
                            continue
                        
                        # L2 misfit: ||d - s||^2 = d^2 + s^2 - 2*d*s
                        
                        # d^2 contribution
                        dd_contrib = data_data[istation, icomp]
                        
                        # s^2 contribution (computed using PETSc)
                        if use_petsc_solvers:
                            # Create matrix for G^T G
                            gg_matrix = PETSc.Mat().create(comm=comm)
                            gg_matrix.setSizes([ngreens, ngreens])
                            gg_matrix.setType('dense')
                            gg_matrix.setUp()
                            
                            # Fill matrix
                            gg_values = greens_greens[istation, icomp, ishift, :, :]
                            for i in range(ngreens):
                                gg_matrix.setValues([i], range(ngreens), gg_values[i, :])
                            
                            gg_matrix.assemblyBegin()
                            gg_matrix.assemblyEnd()
                            
                            # Compute s^2 = source^T * G^T*G * source
                            gg_matrix.mult(source_vec, temp_vec)
                            ss_contrib = source_vec.dot(temp_vec)
                            
                            gg_matrix.destroy()
                        else:
                            # Use NumPy for matrix-vector products
                            gg_values = greens_greens[istation, icomp, ishift, :, :]
                            ss_contrib = np.dot(np.dot(gg_values, sources[isource, :]), 
                                              sources[isource, :])
                        
                        # -2*d*s contribution
                        gd_values = greens_data[istation, icomp, :, ishift]
                        ds_contrib = -2.0 * np.dot(gd_values, sources[isource, :])
                        
                        # Total misfit for this component
                        comp_misfit = dd_contrib + ss_contrib + ds_contrib
                        
                        if hybrid_norm:
                            comp_misfit = np.sqrt(max(0, comp_misfit))
                        
                        misfit_value += weight * dt * comp_misfit
        
        results[isource, 0] = misfit_value
    
    # Clean up PETSc objects
    source_vec.destroy()
    temp_vec.destroy()
    result_vec.destroy()
    
    return results


def _numpy_to_petsc_vec(array, comm):
    """Convert NumPy array to PETSc vector"""
    n = len(array)
    vec = PETSc.Vec().create(comm=comm)
    vec.setSizes(n)
    vec.setUp()
    vec.setValues(range(n), array)
    vec.assemblyBegin()
    vec.assemblyEnd()
    return vec


def _petsc_vec_to_numpy(vec):
    """Convert PETSc vector to NumPy array"""
    return vec.getArray().copy()


#
# Import utility functions from level2
#
from mtuq.misfit.waveform.level2 import (
    _get_time_sampling,
    _get_padding,
    _get_greens,
    _get_data,
    _get_components,
    _get_weights,
    _get_stations,
    _get_groups,
    _check,
    _to_array,
    _corr_1_2,
    _autocorr_1,
    _autocorr_2,
)

