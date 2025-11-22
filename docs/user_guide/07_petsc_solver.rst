
PETSc-Accelerated Solver
========================

Overview
--------

MTUQ now includes a PETSc-accelerated version of the moment tensor inversion solver. This version leverages the Portable, Extensible Toolkit for Scientific Computation (PETSc) for distributed linear algebra operations, providing improved performance for large-scale inversions.

What is PETSc?
--------------

PETSc is a suite of data structures and routines for the scalable solution of scientific applications modeled by partial differential equations. It provides:

- Distributed arrays and vectors
- Parallel linear algebra operations
- High-performance sparse and dense matrix operations
- Integration with MPI for parallel computing

Benefits
--------

The PETSc-based solver offers several advantages:

1. **Improved Performance**: Efficient parallel linear algebra operations
2. **Scalability**: Better performance on multi-core and cluster systems
3. **Memory Efficiency**: Distributed data structures for large problems
4. **Flexibility**: Can use NumPy operations or PETSc solvers

Installation
------------

To use the PETSc solver, you need to install PETSc and petsc4py:

Using conda (recommended)::

    conda install -c conda-forge petsc petsc4py

Using pip::

    pip install petsc4py

For the main MTUQ installation with PETSc support::

    pip install mtuq[petsc]

Basic Usage
-----------

The PETSc solver has the same API as the standard grid search:

.. code-block:: python

    from mtuq import grid_search_petsc
    from mtuq.misfit import Misfit
    
    # Define misfit function
    misfit = Misfit(
        norm='L2',
        time_shift_min=-2.,
        time_shift_max=+2.,
        normalize=True,
    )
    
    # Run PETSc-accelerated grid search
    results = grid_search_petsc(
        data, greens, misfit, origin, grid,
        use_petsc_solvers=True)

Parameters
----------

The ``grid_search_petsc`` function accepts all the same parameters as ``grid_search``, plus:

``use_petsc_solvers`` : bool, optional (default: True)
    If True, use PETSc KSP solvers for linear algebra operations.
    If False, use standard NumPy operations wrapped in PETSc vectors.

Running with MPI
----------------

The PETSc solver automatically detects and uses MPI for parallel execution:

Single process::

    python GridSearch.DoubleCouple.PETSc.py

Multiple processes::

    mpirun -n 4 python GridSearch.DoubleCouple.PETSc.py

The solver will automatically distribute the work across available processes.

Example
-------

A complete example is provided in ``examples/GridSearch.DoubleCouple.PETSc.py``:

.. code-block:: python

    #!/usr/bin/env python
    
    from mtuq import read, download_greens
    from mtuq.event import Origin
    from mtuq.grid import DoubleCoupleGridRegular
    from mtuq.grid_search_petsc import grid_search_petsc
    from mtuq.misfit import Misfit
    
    # Check PETSc availability
    try:
        from petsc4py import PETSc
        print("PETSc version:", PETSc.Sys.getVersion())
    except ImportError:
        print("PETSc not available")
        exit(1)
    
    # Initialize PETSc
    if not PETSc.Sys.isInitialized():
        PETSc.Sys.init()
    
    # ... setup data, greens, misfit, grid ...
    
    # Run PETSc-accelerated inversion
    results = grid_search_petsc(
        data, greens, misfit, origin, grid,
        use_petsc_solvers=True)

Performance Considerations
--------------------------

1. **Small Problems**: For small inversions (< 1000 sources), the overhead of PETSc may not provide benefits. Use the standard solver in these cases.

2. **Large Problems**: For large inversions (> 10,000 sources), PETSc can provide significant speedups, especially with multiple MPI processes.

3. **Memory**: PETSc uses distributed arrays, which can help with memory-intensive problems.

4. **Solver Choice**: The ``use_petsc_solvers`` parameter allows you to choose between:
   
   - ``True``: Full PETSc solvers (more overhead, better for very large problems)
   - ``False``: NumPy operations with PETSc data structures (lower overhead)

Comparison with Standard Solver
--------------------------------

The PETSc solver produces identical results to the standard solver but with different performance characteristics:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Solver
     - Small Grids
     - Large Grids
     - MPI Support
     - Memory Usage
   * - Standard
     - Fast
     - Moderate
     - Yes (mpi4py)
     - Higher
   * - PETSc
     - Moderate
     - Fast
     - Yes (built-in)
     - Lower

Troubleshooting
---------------

**Import Error: PETSc not available**

Install PETSc and petsc4py::

    conda install -c conda-forge petsc4py

**MPI Version Conflicts**

If you encounter MPI version conflicts, ensure that petsc4py is compiled with the same MPI as mpi4py::

    conda install -c conda-forge petsc4py mpi4py

**Performance Not Improving**

Try adjusting the ``use_petsc_solvers`` parameter or ensure you're running with multiple MPI processes.

Technical Details
-----------------

The PETSc solver implements the same L2 misfit computation as the standard solver but uses:

1. **PETSc Vectors** for source parameters and data
2. **PETSc Matrices** for Green's function tensors
3. **Parallel matrix-vector products** for computing synthetics
4. **Distributed reduction operations** for summing misfit contributions

The implementation is in:

- ``mtuq/grid_search_petsc.py`` - Main grid search with PETSc
- ``mtuq/misfit/waveform/level_petsc.py`` - PETSc misfit computation

References
----------

- PETSc documentation: https://petsc.org/release/
- petsc4py documentation: https://petsc.org/release/petsc4py/
- MTUQ documentation: https://uafgeotools.github.io/mtuq/

