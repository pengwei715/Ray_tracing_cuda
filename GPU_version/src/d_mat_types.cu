#include "d_mat_types.h"


__device__ float getElement(const d_MatFloat A, int row, int col){
  return A.data[row*A.stride + col];
}
__device__ void setElement(d_MatFloat A, int row, int col, float val){
  A.data[row*A.stride + col] = val;
}
