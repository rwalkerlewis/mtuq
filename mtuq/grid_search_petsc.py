"""
Grid search using PETSc for distributed linear algebra

This module provides a PETSc-accelerated version of the moment tensor 
inversion grid search, leveraging PETSc's distributed arrays and parallel
linear algebra operations for improved performance.
"""

import numpy as np
import pandas
import xarray

from collections.abc import Iterable
from mtuq.event import Origin
from mtuq.grid import DataFrame, DataArray, Grid, UnstructuredGrid
from mtuq.util import gather2, iterable, timer, remove_list, warn,\
    ProgressCallback, dataarray_idxmin, dataarray_idxmax
from os.path import splitext
from xarray.core.formatting import unindexed_dims_repr

try:
    from petsc4py import PETSc
    PETSC_AVAILABLE = True
except ImportError:
    PETSC_AVAILABLE = False
    PETSc = None


xarray.set_options(keep_attrs=True)


def grid_search_petsc(data, greens, misfit, origins, sources, 
    msg_interval=25, timed=True, verbose=1, gather=True, 
    use_petsc_solvers=True):

    """ Evaluates misfit over grids using PETSc for computation

    .. rubric :: Usage

    Carries out a grid search by evaluating 
    `misfit(data, greens.select(origin), source)` over all origins and sources,
    using PETSc distributed arrays and parallel linear algebra operations.

    If `origins` and `sources` are regularly-spaced, returns an `MTUQDataArray`
    containing misfit values and corresponding grid points. Otherwise, 
    an `MTUQDataFrame` is returned.


    .. rubric :: Input arguments

    ``data`` (`mtuq.Dataset`):
    The observed data to be compared with synthetic data


    ``greens`` (`mtuq.GreensTensorList`):
    Green's functions used to generate synthetic data


    ``misfit`` (`mtuq.Misfit` or some other function):
    Misfit function


    ``origins`` (`list` of `mtuq.Origin` objects):
    Origins to be searched over


    ``sources`` (`mtuq.Grid` or `mtuq.UnstructuredGrid`):
    Source mechanisms to be searched over


    ``msg_interval`` (`int`):
    How frequently, as a percentage of total evaluations, should progress 
    messages be displayed? (`int` between 0 and 100)


    ``timed`` (`bool`):
    Displays elapsed time at end


    ``gather`` (`bool`):
    If `True`, process 0 returns all results and any other processes return
    `None`.  Otherwise, results are divided evenly among processes.
    (ignored outside MPI environment)


    ``use_petsc_solvers`` (`bool`):
    If `True`, use PETSc KSP solvers for linear algebra operations.
    If `False`, use standard NumPy operations wrapped in PETSc vectors.


    .. note:

      This function uses PETSc for distributed linear algebra operations.
      If invoked from an MPI environment, PETSc will automatically distribute
      work across processes. The underlying computation uses PETSc vectors
      and matrices for efficient parallel operations.

    """
    if not PETSC_AVAILABLE:
        raise ImportError(
            "PETSc not available. Please install petsc4py:\n"
            "  conda install -c conda-forge petsc4py\n"
            "or\n"
            "  pip install petsc4py"
        )

    # check grid
    origins = iterable(origins)
    for origin in origins:
        assert type(origin) is Origin

    if type(sources) not in (Grid, UnstructuredGrid):
        raise TypeError

    size = len(origins)*sources.size

    # initialize PETSc
    if not PETSc.Sys.isInitialized():
        PETSc.Sys.init()

    comm = PETSc.COMM_WORLD
    iproc = comm.rank
    nproc = comm.size

    # print debugging information
    if verbose>0 and iproc==0:
        print('  Using PETSc-based grid search')
        print('  PETSc version:', PETSc.Sys.getVersion())
        print()
        
        try:
            print(misfit.description())
        except:
            pass

        print('    Number of misfit evaluations: {:,}\n'.format(size))
        
        if nproc > 1:
            print('    Number of PETSc processes: {:,}'.format(nproc))
            print('    Number of evaluations per process: {:,}\n'.format(size//nproc))


    if nproc > 1:
        #
        # divide up the grid search over PETSc processes
        #
        if nproc > sources.size:
            raise Exception('Number of CPU cores exceeds size of grid')

        _all = sources
        _subsets = None
        if iproc == 0:
            _subsets = sources.partition(nproc)
        sources = comm.bcast(_subsets, iproc)[iproc] if iproc == 0 else comm.bcast(None, 0)[iproc]

        if iproc != 0:
            timed = False
            msg_interval = 0


    #
    # evaluate misfit over grids using PETSc
    #
    values = _grid_search_petsc_serial(
        data, greens, misfit, origins, sources, timed=timed,
        msg_interval=msg_interval, use_petsc_solvers=use_petsc_solvers)


    #
    # collect results
    #
    if nproc > 1 and gather:
        # gather results from PETSc processes
        values_list = comm.gather(values, root=0)
        
        if iproc == 0:
            values = np.concatenate(values_list, axis=0)
            sources = _all
        else:
            return

    # convert from NumPy array to DataArray or DataFrame
    if issubclass(type(sources), Grid):
        return _to_dataarray(origins, sources, values)

    elif issubclass(type(sources), UnstructuredGrid):
        return _to_dataframe(origins, sources, values)


@timer
def _grid_search_petsc_serial(data, greens, misfit, origins, sources, 
    timed=True, msg_interval=25, use_petsc_solvers=True):
    """ Evaluates misfit over origin and source grids 
    (PETSc-accelerated serial implementation)
    """
    from mtuq.misfit.waveform import level_petsc
    
    ni = len(origins)
    nj = len(sources)

    values = []
    for _i, origin in enumerate(origins):

        msg_handle = ProgressCallback(
            start=_i*nj, stop=ni*nj, percent=msg_interval)

        # evaluate misfit function using PETSc
        if hasattr(misfit, 'norm'):
            # Use PETSc-accelerated misfit computation
            values += [level_petsc.misfit(
                data, greens.select(origin), sources, 
                misfit.norm, misfit.time_shift_groups,
                misfit.time_shift_min, misfit.time_shift_max, 
                msg_handle, normalize=misfit.normalize,
                use_petsc_solvers=use_petsc_solvers)]
        else:
            # Fall back to standard misfit evaluation
            values += [misfit(
                data, greens.select(origin), sources, msg_handle)]

    # returns NumPy array of shape `(len(sources), len(origins))` 
    return np.concatenate(values, axis=1)


#
# Import data structures from main grid_search module
#
from mtuq.grid_search import (
    MTUQDataArray, MTUQDataFrame,
    _to_dataarray, _to_dataframe,
    open_ds, open_da, open_df
)

