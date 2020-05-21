#ifndef D_MAT_TYPES_H
#define D_MAT_TYPES_H

#include <stdlib.h>
#include <string.h>
#include <complex.h>

// **********************************************
// For floats
// **********************************************

typedef struct {
  size_t m, n;
  float *data;
  size_t stride;
} d_MatFloat;

__device__ float getElement(const d_MatFloat A, int row, int col);

__device__ void setElement(d_MatFloat A, int row, int col, float val);

#endif
