#ifndef MPCS_HPC_SRC_MPCS_HPC_MAT_TYPES_H
#define MPCS_HPC_SRC_MPCS_HPC_MAT_TYPES_H

#include <stdlib.h>
#include <string.h>
#include <complex.h>

// **********************************************
// For ints
// **********************************************

typedef struct {
  size_t m, n;
  int **at;
  int *data;
} MatInt;

MatInt mat_malloc_int(size_t m, size_t n);

void mat_free_int(MatInt *mat);

// **********************************************
// For floats
// **********************************************

typedef struct {
  size_t m, n;
  float **at;
  float *data;
} MatFloat;

MatFloat mat_malloc_float(size_t m, size_t n);

void mat_free_float(MatFloat *mat);

// **********************************************
// For doubles
// **********************************************

typedef struct {
  size_t m, n;
  double **at;
  double *data;
} MatDouble;

MatDouble mat_malloc_double(size_t m, size_t n);

void mat_free_double(MatDouble *mat);

// **********************************************
// For complex
// **********************************************

typedef struct {
  size_t m, n;
  complex **at;
  complex *data;
} MatComplex;

MatComplex mat_malloc_complex(size_t m, size_t n);

void mat_free_complex(MatComplex *mat);

#endif // MPCS_HPC_SRC_MPCS_HPC_MAT_TYPES_H
