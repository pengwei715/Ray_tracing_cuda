#include "vec.h"

Vec vec_add(Vec v1, Vec v2) {
  Vec res;
  res.x = v1.x + v2.x;
  res.y = v1.y + v2.y;
  res.z = v1.z + v2.z;
  return res;
}

Vec vec_revert(Vec v) {
  Vec res;
  res.x = -v.x;
  res.y = -v.y;
  res.z = -v.z;
  return res;
}

Vec vec_sub(Vec v1, Vec v2) {
  Vec res;
  res.x = v1.x - v2.x;
  res.y = v1.y - v2.y;
  res.z = v1.z - v2.z;
  return res;
}

Vec vec_scale(Vec v, double i) {
  Vec res;
  res.x = i * v.x;
  res.y = i * v.y;
  res.z = i * v.z;
  return res;
}

double rand_double(uint64_t *seed){
  const uint64_t m = 9223372036854775808ULL;
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) /(double) m;
}

Vec vec_sample_unit(int i) {
  Vec res;
  //  uint64_t seed = (uint64_t)(i * 4238811);
  double rd1 = (double)rand() / (double)RAND_MAX;
  double rd2 = (double)rand() / (double)RAND_MAX;
  //double pho = rand_double(&seed) * 2 * M_PI;			   
  //double cos_theta = rand_double(&seed) * 2 - 1;
  double pho = rd1 * 2 * M_PI;
  double cos_theta = rd2 * 2 - 1;
  double sin_theta = sqrt( 1 - cos_theta * cos_theta);
  res.x = sin_theta * cos(pho);
  res.y = sin_theta * sin(pho);
  res.z = cos_theta;
  return res;
}

double vec_dot_product(Vec v1, Vec v2) {
  double res;
  res = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  return res;
}

double vec_norm(Vec v) {
  double res;
  res = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  return res;
}
