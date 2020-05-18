#include "mat_types.h"

MatInt mat_malloc_int(size_t m, size_t n) {
  MatInt mat;
  mat.m = m;
  mat.n = n;
  posix_memalign((void *)&mat.at, 64, m * sizeof(int *));
  posix_memalign((void *)&mat.data, 64, m * n * sizeof(int));
  memset(mat.data, 0, m * n * sizeof(int));

  for (size_t i = 0; i < m; ++i) {
    mat.at[i] = &(mat.data[i * n]);
  }

  return mat;
}

void mat_free_int(MatInt *mat) {
  free(mat->data);
  free(mat->at);
}

MatFloat mat_malloc_float(size_t m, size_t n) {
  MatFloat mat;
  mat.m = m;
  mat.n = n;
  posix_memalign((void *)&mat.at, 64, m * sizeof(float *));
  posix_memalign((void *)&mat.data, 64, m * n * sizeof(float));
  memset(mat.data, 0, m * n * sizeof(float));

  for (size_t i = 0; i < m; ++i) {
    mat.at[i] = &(mat.data[i * n]);
  }

  return mat;
}

void mat_free_float(MatFloat *mat) {
  free(mat->data);
  free(mat->at);
}

MatDouble mat_malloc_double(size_t m, size_t n) {
  MatDouble mat;
  mat.m = m;
  mat.n = n;
  posix_memalign((void *)&mat.at, 64, m * sizeof(double *));
  posix_memalign((void *)&mat.data, 64, m * n * sizeof(double));
  memset(mat.data, 0, m * n * sizeof(double));

  for (size_t i = 0; i < m; ++i) {
    mat.at[i] = &(mat.data[i * n]);
  }

  return mat;
}

void mat_free_double(MatDouble *mat) {
  free(mat->data);
  free(mat->at);
}

MatComplex mat_malloc_complex(size_t m, size_t n) {
  MatComplex mat;
  mat.m = m;
  mat.n = n;
  posix_memalign((void *) &mat.at, 64, m * sizeof(complex *));
  posix_memalign((void *) &mat.data, 64, m * n * sizeof(complex));
  memset(mat.data, 0, m * n * sizeof(complex));

  for (size_t i = 0; i < m; ++i) {
    mat.at[i] = &(mat.data[i * n]);
  }

  return mat;
}

void mat_free_complex(MatComplex *mat) {
  free(mat->data);
  free(mat->at);
}
