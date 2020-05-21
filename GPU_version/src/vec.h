#ifndef VEC_H
#define VEC_H

#include <stdlib.h>
#include <math.h>
#include <stdint.h>

typedef struct {
  float x;
  float y;
  float z;
} Vec;

__device__ float rand_float(uint64_t *seed);
__device__ Vec vec_add(Vec v1, Vec v2);
__device__ Vec vec_sub(Vec v1, Vec v2);
__device__ Vec vec_scale(Vec v, float i);
__device__ Vec vec_sample_unit(int i);
__device__ float vec_dot_product(Vec v1, Vec v2);
__device__ float vec_norm(Vec v);

#endif
