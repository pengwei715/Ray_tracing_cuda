#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include "vec.h"
#include "d_mat_types.h"

const int  TILE_WIDTH=16;

__device__ void update(float tile_cache[TILE_WIDTH][TILE_WIDTH], 
		       float w_x_min, float w_x_max, 
		       float w_z_min, float w_z_max, int threadnum) {
  Vec C;
  C.x = 0.0;
  C.y = 12.0;
  C.z = 0.0;

  Vec L;
  L.x = 4.0;
  L.y = 4.0;
  L.z = -1;
  float w_y = 10;
  float w_max = TILE_WIDTH/2;
  float R = 6.0;

  Vec V, W;
  while(1){
    V = vec_sample_unit(threadnum);
    if (V.y == 0) continue;
    W = vec_scale(V, (w_y/V.y));
    float temp = vec_dot_product(V, C);
    if (W.x < w_x_max && W.x > w_x_min && 
	W.z < w_z_max && W.z > w_z_min && 
	temp * temp + R * R - vec_dot_product(C,C) > 0) {
      break;
    }
  }
  float temp2 = vec_dot_product(V,C);
  float t = temp2 - sqrt(temp2 * temp2 + R * R - vec_dot_product(C,C));
  Vec II = vec_scale(V, t);
  Vec N = vec_scale(vec_sub(II,C), 1.0/vec_norm(vec_sub(II,C)));
  Vec S = vec_scale(vec_sub(L,II), 1.0/vec_norm(vec_sub(L,II)));
  float b = vec_dot_product(S,N); 
  b = 0 > b ? 0 : b;
  int x = floor((W.x + w_max) /(2 * w_max) * (TILE_WIDTH - 1));
  int z = floor((W.z + w_max) /(2 * w_max) * (TILE_WIDTH - 1));
  atomicAdd(&tile_cache[x][z], b);
}

__global__ void ray_trace_kernel(d_MatFloat d_window, int width, long ray_num) {
  // init the shared mem cache
  __shared__ float tile_cache[TILE_WIDTH][TILE_WIDTH];
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  //init the tile element to 0
  for (int i = 0; i < TILE_WIDTH/blockDim.x; ++i) {
    for (int j = 0; j < TILE_WIDTH/blockDim.y; ++j) {
      if (tx + i < TILE_WIDTH && ty + j < TILE_WIDTH){
	tile_cache[tx + i][ty + j] = 0;
      }
    }
  }
  __syncthreads();
  
  //issue the threads to do the work, each thread has ray_num/(gridDim.x * gridDim.y * blockDim.x * blockDim.y)
  int num = ray_num/(gridDim.x * gridDim.y * blockDim.x * blockDim.y);
  float w_x_min = blockIdx.x * TILE_WIDTH;
  float w_z_min = blockIdx.y * TILE_WIDTH;
  float w_x_max = blockIdx.x * (TILE_WIDTH + 1);
  float w_z_max = blockIdx.y * (TILE_WIDTH + 1);

  for (int i = 0; i < num; ++i){
    update(tile_cache, w_x_min, w_x_max, w_z_min, w_z_max, (blockIdx.x+blockIdx.y) * TILE_WIDTH + tx + ty);
  }
  __syncthreads();

  //copy the result back to global memory
  for (int i = 0; i < TILE_WIDTH/blockDim.x; ++i) {
    for (int j = 0; j < TILE_WIDTH/blockDim.y; ++j) {
      if (tx + i < TILE_WIDTH && ty + j < TILE_WIDTH){
	float b = tile_cache[tx + i][ty + j];
	int row = blockIdx.x * TILE_WIDTH + tx + i;
	int col = blockIdx.y * TILE_WIDTH + ty + j;
	setElement(d_window, row, col, b);
      }
    }
  }
  __syncthreads();
}



void get_input(int argc, char *argv[], long* num_rays, size_t* grid_len){
  *num_rays = 5e6;
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
  long num_rays;
  size_t grid_len;
  get_input(argc, argv, &num_rays, &grid_len);
  
  d_MatFloat window, d_window;
  window.m = grid_len;
  window.n = grid_len;
  window.stride = grid_len;
  window.data = (float*) calloc(0.0, grid_len*grid_len*sizeof(float));
  d_window.m = grid_len;
  d_window.n = grid_len;
  d_window.stride = grid_len;
  
  size_t size = grid_len * grid_len * sizeof(float);
  cudaMalloc(&d_window.data, size);
  cudaMemcpy(d_window.data, window.data, size, cudaMemcpyHostToDevice);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid(window.m/dimBlock.x, window.n/dimBlock.y);
  
  ray_trace_kernel<<<dimGrid, dimBlock>>>(d_window, grid_len, num_rays);
  
  cudaMemcpy(window.data, d_window.data, size, cudaMemcpyDeviceToHost);
  cudaFree(d_window.data);

  char filename[] = "ball.dat";
  FILE *f = fopen(filename, "wb");
  if (f != NULL) {
    fwrite(window.data, sizeof(float), window.m * window.n, f);
    fclose(f);
  } else {
    fprintf(stderr, "Error opening %s: ", filename);
    perror("");
    free(window.data);
    exit(EXIT_FAILURE);
  }
  free(window.data);
  return 0;
}




