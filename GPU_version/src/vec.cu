#include "vec.h"

__device__ Vec vec_add(Vec v1, Vec v2) {
  Vec res;
  res.x = v1.x + v2.x;
  res.y = v1.y + v2.y;
  res.z = v1.z + v2.z;
  return res;
}

__device__ Vec vec_sub(Vec v1, Vec v2) {
  Vec res;
  res.x = v1.x - v2.x;
  res.y = v1.y - v2.y;
  res.z = v1.z - v2.z;
  return res;
}

__device__ Vec vec_scale(Vec v, float i) {
  Vec res;
  res.x = i * v.x;
  res.y = i * v.y;
  res.z = i * v.z;
  return res;
}


__device__ float rand_float(uint64_t *seed){
  const uint64_t m = 9223372036854775808ULL;
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (float)(*seed) / (float) m;
}

__device__ Vec vec_sample_unit(int i) {
  Vec res;
  uint64_t seed = ((uint64_t)(i * 4238811));

  float pho = rand_float(&seed) * 2 * M_PI;			   
  float cos_theta = rand_float(&seed) * 2 - 1;

  float sin_theta = sqrt( 1 - cos_theta * cos_theta);
  res.x = sin_theta * cos(pho);
  res.y = sin_theta * sin(pho);
  res.z = cos_theta;
  return res;
}

__device__ float vec_dot_product(Vec v1, Vec v2) {
  float res;
  res = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  return res;
}

__device__ float vec_norm(Vec v) {
  float res;
  res = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  return res;
}
