
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <math.h>
#include <petscvec.h>
#include <petscmat.h>

//
// array access macros
//
#define data_data(i0,i1)\
    (*(npy_float64*)((PyArray_DATA(data_data)+\
    (i0) * PyArray_STRIDES(data_data)[0]+\
    (i1) * PyArray_STRIDES(data_data)[1])))

#define greens_data(i0,i1,i2,i3)\
    (*(npy_float64*)((PyArray_DATA(greens_data)+\
    (i0) * PyArray_STRIDES(greens_data)[0]+\
    (i1) * PyArray_STRIDES(greens_data)[1]+\
    (i2) * PyArray_STRIDES(greens_data)[2]+\
    (i3) * PyArray_STRIDES(greens_data)[3])))

#define greens_greens(i0,i1,i2,i3,i4)\
    (*(npy_float64*)((PyArray_DATA(greens_greens)+\
    (i0) * PyArray_STRIDES(greens_greens)[0]+\
    (i1) * PyArray_STRIDES(greens_greens)[1]+\
    (i2) * PyArray_STRIDES(greens_greens)[2]+\
    (i3) * PyArray_STRIDES(greens_greens)[3]+\
    (i4) * PyArray_STRIDES(greens_greens)[4])))

#define sources(i0,i1)\
    (*(npy_float64*)((PyArray_DATA(sources)+\
    (i0) * PyArray_STRIDES(sources)[0]+\
    (i1) * PyArray_STRIDES(sources)[1])))

#define groups(i0,i1)\
    (*(npy_float64*)((PyArray_DATA(groups)+\
    (i0) * PyArray_STRIDES(groups)[0]+\
    (i1) * PyArray_STRIDES(groups)[1])))

#define weights(i0,i1)\
    (*(npy_float64*)((PyArray_DATA(weights)+\
    (i0) * PyArray_STRIDES(weights)[0]+\
    (i1) * PyArray_STRIDES(weights)[1])))

#define results(i0)\
    (*(npy_float64*)((PyArray_DATA(results)+\
    (i0) * PyArray_STRIDES(results)[0])))

#define cc(i0)\
    (*(npy_float64*)((PyArray_DATA(cc)+\
    (i0) * PyArray_STRIDES(cc)[0])))


// Norm selection
typedef enum { NORM_L2, NORM_L1, NORM_HYBRID } NT;

// misfit function
static PyObject *misfit(PyObject *self, PyObject *args) {

   // cross-correlation input arrays
  PyArrayObject *data_data, *greens_data, *greens_greens;

  // other input arrays
  PyArrayObject *sources, *groups, *weights;

  // scalar input arguments
  int hybrid_norm;
  npy_float64 dt;
  int NPAD1, NPAD2;
  int debug_level;
  int msg_start, msg_stop, msg_percent;

  int NSRC, NSTA, NC, NG, NGRP, NPAD;

  // Create PETSc scalars
  PetscInt ista, ic, it, igrp;
  PetscScalar value, weight_value;

  // Create PETSc vectors/matrices
  Vec S, d_d, misfit, final_misfit;
  Mat G_G, G_d, G_d_w, G_d_w_grp, gg, ww, CC;







  // int cc_argmax, j1, j2, nd, NPAD;
  // npy_float64 cc_max, L2_sum, L2_tmp;

  // float iter, next_iter;
  // int msg_count, msg_interval;


  // parse arguments
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!idiiiiii",
                        &PyArray_Type, &data_data,
                        &PyArray_Type, &greens_data,
                        &PyArray_Type, &greens_greens,
                        &PyArray_Type, &sources,
                        &PyArray_Type, &groups,
                        &PyArray_Type, &weights,
                        &hybrid_norm,
                        &dt,
                        &NPAD1,
                        &NPAD2,
                        &debug_level,
                        &msg_start,
                        &msg_stop,
                        &msg_percent)) {
    return NULL;
  }

  // Set norm type
  NT norm_type = (hybrid_norm == 1) ? NORM_HYBRID : (hybrid_norm == 2) ? NORM_L1 : NORM_L2;

  // Initialize PETSc
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(NULL, NULL, NULL, NULL));


  NSRC = (int) PyArray_SHAPE(sources)[0];
  NSTA = (int) PyArray_SHAPE(weights)[0];
  NC = (int) PyArray_SHAPE(weights)[1];
  NG = (int) PyArray_SHAPE(sources)[1];
  NGRP = (int) PyArray_SHAPE(groups)[0];

  NPAD = (int) NPAD1+NPAD2+1;

  if (debug_level>1) {
    printf(" number of sources:  %d\n", NSRC);
    printf(" number of stations:  %d\n", NSTA);
    printf(" number of components:  %d\n", NC);
    printf(" number of Green's functions:  %d\n\n", NG);
    printf(" number of component groups:  %d\n", NGRP);
  }

  // // Convert NumPy arrays to raw data pointers
  PetscScalar *d_d_data = (PetscScalar *) PyArray_DATA(data_data);
  PetscScalar *G_d_data = (PetscScalar *) PyArray_DATA(greens_data);
  PetscScalar *G_G_data = (PetscScalar *) PyArray_DATA(greens_greens);
  PetscScalar *S_data = (PetscScalar *) PyArray_DATA(sources);
  PetscScalar *W_data = (PetscScalar *) PyArray_DATA(weights);
  PetscScalar *Q_data = (PetscScalar *) PyArray_DATA(groups);


  // Assign to PETSc structures
  // PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, NSRC * NG, PETSC_DECIDE, S_data, &S));
  PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, NSTA * NC, PETSC_DECIDE, d_d_data, &d_d));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, NSTA * NC * NG, NG, G_G_data, &G_G));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, NSTA * NC * NPAD, NG, G_d_data, &G_d));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, NSRC, NG, S_data, &s));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, NSRC * NPAD, &cc));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, NSRC, &misfit));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, NSRC, &final_misfit));

  // Create a PetscMat to populate weights
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 1, NSTA*NC*NPAD, &ww));

  // Populate ww matrix
  for (ista = 0; ista < NSTA; ista++) {
    for (ic = 0; ic < NC; ic++) {
        for (it = 0; it < NPAD; it++) {
            col = (ista * NC * NPAD) + (ic * NPAD) + it;
            weight_value = weights[ic][ista];  // Get weight from weights[nc, nsta]
            PetscCall(MatSetValue(W, row, col, weight_value, INSERT_VALUES));
        }
    }
  }

  // Multiply to assign weights
  PetscCall(MatMatMult(G_d, ww, MAT_INITIAL_MATRIX, PETSC_DECIDE, G_d_w));

  // Create a PetscMat to use to zero components not in group
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 1, NSTA*NC*NPAD, &gg));

  // Perform action on time shift groups
  for (igrp = 0; igrp < NGRP; igrp++) {
 
    // Zero out components not included in this group - Maybe just pull out sparse matrix without those values?
    // Populate the vector
    for (ista = 0; ista < NSTA; ista++) {
      for (ic = 0; ic < NC; ic++) {
        // Check if this component is zeroed out based on the group matrix
        PetscBool is_zero = PETSC_TRUE;
        if (groups[igrp][ic] != 0) {
          is_zero = PETSC_FALSE;
          break;
        }
        for (it = 0; it < NPAD; it++) {
          PetscInt idx = (ista * NC * NPAD) + (ic * NPAD) + it;
          value = is_zero ? 0.0 : 1.0;  // Assign 0 if it should be zeroed, otherwise 1.0
          PetscCall(MatSetValue(gg, 0, idx, value, INSERT_VALUES));
        }
      }
    } 
    // Finalize assembly of the vector
    PetscCall(VecAssemblyBegin(gg));
    PetscCall(VecAssemblyEnd(gg));

    // Multiply to zero components not in group
    PetscCall(MatMatMult(G_d_w, gg, MAT_INITIAL_MATRIX, PETSC_DECIDE, G_d_w_grp));

    // Step 1: Compute cross-correlation CC[NSTA*NC*NPAD,NSRC] = G_d[NSTA*NC*NPAD,NG] * S[NG,NSRC] 
    PetscCall(MatMatMult(G_d_w_grp, S, MAT_INITIAL_MATRIX, PETSC_DECIDE, CC));
  
    // Step 2: Find optimal time shift for each source

    // Unravel CC to find optimal NPAD value for each station and component.


    PetscInt its[NSRC];
    PetscCall(VecStrideMax(cc, NPTS, NULL, its));

    // Step 3: Extract the corresponding rows from G_G
    Mat G_G_opt;
    PetscCall(MatGetSubMatrix(G_G, its, NULL, MAT_INITIAL_MATRIX, &G_G_opt));

    // Step 4: Compute misfit
    Vec S_GG_S, S_Gd;
    PetscCall(VecDuplicate(S, &S_GG_S));
    PetscCall(VecDuplicate(S, &S_Gd));

    PetscCall(MatMult(G_G_opt, S, S_GG_S));
    PetscCall(MatMult(G_d, S, S_Gd));
    PetscCall(VecScale(S_Gd, -2.0));
    PetscCall(VecWAXPY(misfit, 1.0, S_GG_S, d_d));
    PetscCall(VecAXPY(misfit, 1.0, S_Gd));


    // Step 5: Apply norm selection
    if (norm_type == NORM_L1) {
        PetscCall(VecAbs(misfit); // L1 norm
    } else if (norm_type == NORM_HYBRID) {
        PetscCall(VecSqrtAbs(misfit); // Hybrid L1-L2 norm
    } else {
        // L2 norm (default)
        // No additional operation needed since we directly computed the squared residuals.
    }
  }
  
  // Step 6: Combine into final vector.
  PetscCall(VecAXPY(final_misfit, 1.0, misfit));


  // Convert PETSc misfit vector to NumPy array
  double *misfit_data;
  PetscCall(VecGetArray(misfit, &misfit_data));
  npy_intp dims[1] = {NSRC};
  PyObject *result = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, misfit_data);

  // Cleanup PETSc objects
  PetscCall(VecRestoreArray(misfit, &misfit_data));
  PetscCall(MatDestroy(&G_G));
  PetscCall(MatDestroy(&G_G_opt));
  PetscCall(MatDestroy(&G_d));
  PetscCall(VecDestroy(&S));
  PetscCall(VecDestroy(&d_d));
  PetscCall(VecDestroy(&cc));
  PetscCall(VecDestroy(&misfit));
  PetscCall(VecDestroy(&S_GG_S));
  PetscCall(VecDestroy(&S_Gd));

  PetscCall(PetscFinalize());

  return result;
}

// Register method with Python
static PyMethodDef methods[] = {
  { "misfit", misfit, METH_VARARGS, "Computes waveform misfit (L2, L1, Hybrid)."},
  { NULL, NULL, 0, NULL }
};

// Initialize Python module
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef misfit_module = {
  PyModuleDef_HEAD_INIT, "c_ext_PETSc", "Misfit function (fast PETSc C implementation)", -1, methods,
};
PyMODINIT_FUNC PyInit_c_ext_PETSc(void) {
  Py_Initialize();
  import_array();
  return PyModule_Create(&misfit_module);
}
#else
PyMODINIT_FUNC initc_ext_PETSc(void) {
  (void) Py_InitModule("c_ext_PETSc", methods);
  import_array();
}
#endif