#ifndef VEC_H
#define VEC_H

#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
  double x;
  double y;
  double z;
} Vec;

double rand_double(uint64_t seed);
Vec vec_add(Vec v1, Vec v2);
Vec vec_revert(Vec v);
Vec vec_sub(Vec v1, Vec v2);
Vec vec_scale(Vec v, double i);
Vec vec_sample_unit();
double vec_dot_product(Vec v1, Vec v2);
double vec_norm(Vec v);

#endif
