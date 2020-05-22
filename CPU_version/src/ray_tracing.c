#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include "vec.h"
#include "mat_types.h"
#include "tick.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void get_input(int argc, char *argv[], long* num_rays, size_t* grid_len){
  *num_rays = 10000000;
  *grid_len = 1000;
  int opt;
  while((opt = getopt(argc, argv, "r:g:")) != -1) {
    switch (opt) {
    case 'r':
      *num_rays = atol(optarg);
      break;
    case 'g':
      *grid_len = atol(optarg);
      break;
    default:break;
    }
  }
}

int main(int argc, char* argv[]) {
  timespec_t start = tick();
  int threads = 1;
#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp single
    {
      threads = omp_get_num_threads();
    }
  }
#endif
  long num_rays;
  size_t grid_len;
  double w_y = 10;
  
  Vec C;
  C.x = 0.0;
  C.y = 12.0;
  C.z = 0.0;

  Vec L;
  L.x = 4.0;
  L.y = 4.0;
  L.z = -1;
  
  double R = 6.0;
  get_input(argc, argv, &num_rays, &grid_len);
  MatDouble window = mat_malloc_double(grid_len, grid_len);
  double C_norm = vec_norm(C);
  double w_max = 10;

#pragma omp parallel
  for (int i=0 ; i<num_rays; ++i) {
    Vec V, W;
    while(1){
      V = vec_sample_unit(i);
      if (V.y == 0) continue;
      W = vec_scale(V, (w_y/V.y));
      double temp = vec_dot_product(V, C);
      if (fabs(W.x) < w_max && fabs(W.z) < w_max && temp * temp + R * R - vec_dot_product(C,C) > 0) break;
    }
    double temp2 = vec_dot_product(V,C);
    double t = temp2 - sqrt(temp2 * temp2 + R * R - vec_dot_product(C,C));
    Vec II = vec_scale(V, t);
    Vec N = vec_scale(vec_sub(II,C), 1.0/vec_norm(vec_sub(II,C)));
    Vec S = vec_scale(vec_sub(L,II), 1.0/vec_norm(vec_sub(L,II)));
    double b = vec_dot_product(S,N);
    
    b = 0 > b ? 0 : b;
    int x = floor((W.x + w_max) /(2 * w_max) * (grid_len - 1));
    int z = floor((W.z + w_max) /(2 * w_max) * (grid_len - 1));
#pragma omp critical
    window.at[x][z] += b;
  }
  timespec_t end = tick();
  long dur = tick_diff(start, end);
  printf("%d\n", dur/1000000);
  

  char filename[] = "ball.dat";
  FILE *f = fopen(filename, "wb");
  if (f != NULL) {
    fwrite(window.data, sizeof(double), window.m * window.n, f);
    fclose(f);
  } else {
    fprintf(stderr, "Error opening %s: ", filename);
    perror("");
    mat_free_double(&window);
    exit(EXIT_FAILURE);
  }

  mat_free_double(&window);
  return 0;
}




