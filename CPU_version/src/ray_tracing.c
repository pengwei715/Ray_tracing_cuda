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
  *num_rays = 100000;
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
  double w_y = 1.2;
  double w_max = 2.4;
  
  Vec C;
  C.x = 0.0;
  C.y = 4.0;
  C.z = 4.0;

  Vec L;
  L.x = 2.0;
  L.y = -4.0;
  L.z = 2.0;
  
  double R = 1.0;
  get_input(argc, argv, &num_rays, &grid_len);
  MatDouble window = mat_malloc_double(grid_len, grid_len);
  double C_norm = vec_norm(C);

  timespec_t start = tick();
#pragma omp parallel for shared(window)
  for (int i=0 ; i<num_rays; ++i) {
    Vec V, W;
    while(1){
      V = vec_sample_unit();
      if (V.y == 0) continue;
      W = vec_scale(V, (w_y/V.y));
      double temp = vec_dot_product(V, C);
      if (abs(W.x) < w_max && abs(W.z) < w_max && temp * temp + R * R - C_norm > 0) break;
    }
    double temp2 = vec_dot_product(V,C);
    double t = temp2 - sqrt(temp2 * temp2 + R * R - C_norm);
    Vec II = vec_scale(V, t);
    Vec N = vec_scale(vec_sub(II,C), vec_norm(vec_sub(II,C)));
    Vec S = vec_scale(vec_sub(L,II), vec_norm(vec_sub(L,II)));
    double b = vec_dot_product(S,N);
    b = 0 > b ? 0 : b;
    int x = floor(grid_len/2 - W.x);
    int z = floor(grid_len/2 + W.z);
#pragma omp atomic
    window.at[x][z] += b;      
  }
  timespec_t end = tick();
  long dur = tick_diff(start, end);
  

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




