
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <math.h>
#include <petsc.h>


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
typedef enum { NORM_L2, NORM_L1, NORM_HYBRID } NormType;

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

  int NSRC, NSTA, NC, NG, NGRP;
  int isrc, ista, ic, ig, igrp;

  int cc_argmax, it, itpad, j1, j2, nd, NPAD;
  npy_float64 cc_max, L2_sum, L2_tmp;

  float iter, next_iter;
  int msg_count, msg_interval;


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
  NormType norm_type = (hybrid_norm == 1) ? NORM_HYBRID : (hybrid_norm == 2) ? NORM_L1 : NORM_L2;

  // Initialize PETSc
  PetscErrorCode ierr;
  ierr = PetscInitialize(NULL, NULL, NULL, NULL); CHKERRQ(ierr);


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

  // Convert NumPy arrays to raw data pointers
  double *d_d_data = (double *) PyArray_DATA(data_data);
  double *G_d_data = (double *) PyArray_DATA(greens_data);
  double *G_G_data = (double *) PyArray_DATA(greens_greens);
  double *S_data = (double *) PyArray_DATA(sources);
  double *W_data = (double *) PyArray_DATA(weights);
  double *Q_data = (double *) PyArray_DATA(groups);


  // Create PETSc vectors/matrices
  Vec S, d_d, misfit, cc;
  Mat G_G, G_d;

  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, NSRC * NG, PETSC_DECIDE, S_data, &S); CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, NSTA * NC, PETSC_DECIDE, d_d_data, &d_d); CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, NSTA * NC * NG, NG, G_G_data, &G_G); CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, NSTA * NC * NG, NPTS, G_d_data, &G_d); CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, NSRC * NPTS, &cc); CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, NSRC, &misfit); CHKERRQ(ierr);


  // Step 1: Compute cross-correlation cc = G_d * S
  ierr = MatMult(G_d, S, cc); CHKERRQ(ierr);

  // Step 2: Find optimal time shift for each source
  PetscInt its[NSRC];
  ierr = VecStrideMax(cc, NPTS, NULL, its); CHKERRQ(ierr);

  // Step 3: Extract the corresponding rows from G_G
  Mat G_G_opt;
  ierr = MatGetSubMatrix(G_G, its, NULL, MAT_INITIAL_MATRIX, &G_G_opt); CHKERRQ(ierr);

  // Step 4: Compute misfit
  Vec S_GG_S, S_Gd;
  ierr = VecDuplicate(S, &S_GG_S); CHKERRQ(ierr);
  ierr = VecDuplicate(S, &S_Gd); CHKERRQ(ierr);

  ierr = MatMult(G_G_opt, S, S_GG_S); CHKERRQ(ierr);
  ierr = MatMult(G_d, S, S_Gd); CHKERRQ(ierr);
  ierr = VecScale(S_Gd, -2.0); CHKERRQ(ierr);
  ierr = VecWAXPY(misfit, 1.0, S_GG_S, d_d); CHKERRQ(ierr);
  ierr = VecAXPY(misfit, 1.0, S_Gd); CHKERRQ(ierr);


  // Step 5: Apply norm selection
  if (norm_type == NORM_L1) {
      ierr = VecAbs(misfit); CHKERRQ(ierr);  // L1 norm
  } else if (norm_type == NORM_HYBRID) {
      ierr = VecSqrtAbs(misfit); CHKERRQ(ierr);  // Hybrid L1-L2 norm
  } else {
      // L2 norm (default)
      // No additional operation needed since we directly computed the squared residuals.
  }

    // Convert PETSc misfit vector to NumPy array
    double *misfit_data;
    ierr = VecGetArray(misfit, &misfit_data); CHKERRQ(ierr);
    npy_intp dims[1] = {NSRC};
    PyObject *result = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, misfit_data);

    // Cleanup PETSc objects
    ierr = VecRestoreArray(misfit, &misfit_data); CHKERRQ(ierr);
    ierr = MatDestroy(&G_G); CHKERRQ(ierr);
    ierr = MatDestroy(&G_G_opt); CHKERRQ(ierr);
    ierr = MatDestroy(&G_d); CHKERRQ(ierr);
    ierr = VecDestroy(&S); CHKERRQ(ierr);
    ierr = VecDestroy(&d_d); CHKERRQ(ierr);
    ierr = VecDestroy(&cc); CHKERRQ(ierr);
    ierr = VecDestroy(&misfit); CHKERRQ(ierr);
    ierr = VecDestroy(&S_GG_S); CHKERRQ(ierr);
    ierr = VecDestroy(&S_Gd); CHKERRQ(ierr);

    ierr = PetscFinalize(); CHKERRQ(ierr);

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
PyMODINIT_FUNC initc_ext_L2(void) {
  (void) Py_InitModule("c_ext_PETSc", methods);
  import_array();
}
#endif




















  // allocate arrays
  nd = 1;
  npy_intp dims_cc[] = {(int)NPAD};
  PyObject *cc = PyArray_SimpleNew(nd, dims_cc, NPY_DOUBLE);

  nd = 2;
  npy_intp dims_results[] = {(int)NSRC, 1};
  PyObject *results = PyArray_SimpleNew(nd, dims_results, NPY_DOUBLE);


  // initialize progress messages
  if (msg_percent > 0) {
    msg_interval = msg_percent/100.*msg_stop;
    msg_count = 100./msg_percent*msg_start/msg_stop;
    iter = (float) msg_start;
    next_iter = (float) msg_count*msg_interval;

  }
  else {
    iter = 0;
    next_iter = INFINITY;
  }

  //
  // Print the weights for each component of each station, with numbering to keep track of things
  //

  if (debug_level>1) {
    printf("The weights: \n");
    for (ista=0; ista<NSTA; ista++) {
      for (ic=0; ic<NC; ic++) {
        printf("  %d %d %f \n", ista, ic, weights(ista, ic));
      }
    }
  }

  //
  // Iterate over sources
  //

  for(isrc=0; isrc<NSRC; ++isrc) {


    // display progress message
    if (iter >= next_iter) {
        printf("  about %d percent finished\n", msg_percent*msg_count);
        msg_count += 1;
        next_iter = msg_count*msg_interval;
    }
    iter += 1;


    L2_sum = (npy_float64) 0.;

    for (ista=0; ista<NSTA; ista++) {
      for (igrp=0; igrp<NGRP; igrp++) {

        /*

        Finds the shift between data and synthetics that yields the maximum
        cross-correlation value across all components in the given component
        group, subject to the (time_shift_min, time_shift_max) constraint

        */

        for (it=0; it<NPAD; it++) {
          cc(it) = (npy_float64) 0.;
        }

        for (ic=0; ic<NC; ic++) {

          // Skip components not in the component group being considered
          if (((int) groups(igrp,ic))==0) {
            continue;
           }

          // Skip traces that have been assigned zero weight
          if (fabs(weights(ista, ic)) < 1.e-6) {
              if (debug_level>1) {
                if (isrc==0) {
                  printf(" skipping trace: %d %d\n", ista, ic);
                }
              }
              continue;
           }

          // Sum cross-correlations of all components being considered
          for (ig=0; ig<NG; ig++) {
            for (it=0; it<NPAD; it++) {
                cc(it) += greens_data(ista,ic,ig,it) * sources(isrc,ig);
            }
          }
        }
        cc_max = -NPY_INFINITY;
        cc_argmax = 0;
        for (it=0; it<NPAD; it++) {
          if (cc(it) > cc_max) {
            cc_max = cc(it);
            cc_argmax= it;
          }
        }
        itpad = cc_argmax;


        /*

        Calculates L2 norm of difference between data and synthetics
        for all components in the given component group

        Rather than storing (s - d) directly for all time samples, we use a
        computational shortcut based on

        ||s - d||^2 = s^2 + d^2 - 2sd

        */
        for (ic=0; ic<NC; ic++) {
          L2_tmp = 0.;

          // Skip components not in the component group being considered
          if (((int) groups(igrp,ic))==0) {
            continue;
          }

          // Skip traces that have been assigned zero weight
          if (fabs(weights(ista, ic)) < 1.e-6) {
              continue;
          }

          // calculate s^2
          for (j1=0; j1<NG; j1++) {
            for (j2=0; j2<NG; j2++) {
              L2_tmp += sources(isrc, j1) * sources(isrc, j2) *
                  greens_greens(ista,ic,itpad,j1,j2);
            }
          }

          // calculate d^2
          L2_tmp += data_data(ista,ic);

          // calculate sd
          for (ig=0; ig<NG; ig++) {
            L2_tmp -= 2.*greens_data(ista,ic,ig,itpad) * sources(isrc, ig);
          }

          if (hybrid_norm==0) {
              // L2 norm
              // Print the weight value
              if (debug_level>1) {
                if (isrc==0) {
                  printf(" weight: %d %d %f \n", ista, ic, weights(ista, ic));
                }
              }
              L2_sum += dt * weights(ista,ic) * L2_tmp;
          }
          else {
              // hybrid L1-L2 norm
              L2_sum += dt * weights(ista,ic) * pow(L2_tmp, 0.5);
          }
        }

      }
    }
    results(isrc) = L2_sum;

  }
  return results;

}


//
// Boilerplate
//

static PyMethodDef methods[] = {
    { "misfit", misfit, METH_VARARGS, "Misfit function (fast C implementation)."},
    { NULL, NULL, 0, NULL }
  };


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef misfit_module = {
  PyModuleDef_HEAD_INIT,
  "c_ext_L2",
  "Misfit function (fast C implementation)",
  -1,                  /* m_size */
  methods,             /* m_methods */
  };
#endif


#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_c_ext_L2(void) {
  Py_Initialize();
  import_array();
  return PyModule_Create(&misfit_module);
  }
#else
PyMODINIT_FUNC initc_ext_L2(void) {
  (void) Py_InitModule("c_ext_L2", methods);
  import_array();
  }
#endif
