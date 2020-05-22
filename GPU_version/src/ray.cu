#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

// **********************************************
// For floats vector on device
// **********************************************

typedef struct {
  float x;
  float y;
  float z;
} Vec;

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

__device__ float random_float(uint64_t * seed){
  const uint64_t m = 9223372036854775808ULL;
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c)%m;
  float res = (float) (*seed)/(float)m;
  return res;
}
  
__device__ uint64_t forward(uint64_t seed, uint64_t n){
  const uint64_t m = 9223372036854775808ULL;
  uint64_t a = 2806196910506780709ULL;
  uint64_t c = 1ULL;
  n = n % m;
  uint64_t a_new = 1;
  uint64_t c_new = 0;
  while(n>0){
    if(n & 1){
      a_new *= a;
      c_new = c_new *a + c;
    }
    c *= (a + 1);
    a *= a;
    n >>= 1;
  }
  return (a_new * seed + c_new) % m;
}

__device__ Vec vec_sample_unit(uint64_t i) {
  Vec res;
  uint64_t seed = forward(i, i*200);
  float pho = random_float(&seed) * 2 * 3.141592653;   
  float cos_theta = random_float(&seed) * 2 - 1;
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

// **********************************************
// ray_trace_kernel
// **********************************************
inline void cuchk(cudaError_t err){
  if (err != cudaSuccess) {
     printf("%s in %s at line %d\n",
     	cudaGetErrorString(err), __FILE__, __LINE__);
     exit(EXIT_FAILURE);
  }
}

__global__ void ray_trace_kernel(float* d_window, int width, int ray_num) {

  //tried to use the shared mem, but it's too small to store the whole window 1000*1000*4bytes = 4 mb  
  
  //issue the threads to do the work
  __shared__ int ray_chunck;
  __shared__ Vec C;
  __shared__ Vec L;
  __shared__ float w_y;
  __shared__ float w_max;
  __shared__ float R;

  int tx = blockIdx.x * gridDim.x + threadIdx.x;
  ray_chunck = ray_num/gridDim.x;
  C.x = 0.0;
  C.y = 12.0;
  C.z = 0.0;
  L.x = 4.0;
  L.y = 4.0;
  L.z = -1;
  w_y = 10;
  w_max = 10;
  R = 6.0;
  
  for (int i = 0; i < ray_chunck; i+=blockDim.x*gridDim.x){
    Vec V, W;
    uint64_t seed = tx;
    while(1){
      seed += ray_num;
      V = vec_sample_unit(seed);
      if (V.y == 0) continue;
      W = vec_scale(V, (w_y/V.y));
      float temp = vec_dot_product(V, C);
      if (fabs(W.x) < w_max && fabs(W.z) < w_max && temp * temp + R * R - vec_dot_product(C,C) > 0) break;
    }
    float temp2 = vec_dot_product(V,C);
    float t = temp2 - sqrt(temp2 * temp2 + R * R - vec_dot_product(C,C));
    Vec II = vec_scale(V, t);
    Vec N = vec_scale(vec_sub(II,C), 1.0/vec_norm(vec_sub(II,C)));
    Vec S = vec_scale(vec_sub(L,II), 1.0/vec_norm(vec_sub(L,II)));
    float b = vec_dot_product(S,N); 
    b = 0 >= b ? 0 : b;
    int x = floor((W.x + w_max) /(2 * w_max) * (width - 1));
    int z = floor((W.z + w_max) /(2 * w_max) * (width - 1));
    atomicAdd(&d_window[x*width + z], b);
  }
}

// **********************************************
// get the args
// **********************************************

void get_input(int argc, char *argv[], int* num_rays, int* len, int *grid_dim, int *block_dim){
  *num_rays = 1000000;
  *len = 1000;
  *grid_dim = -1;
  *block_dim = 256;
  int opt;
  while((opt = getopt(argc, argv, "r:l:g:b:")) != -1) {
    switch (opt) {
    case 'r':
      *num_rays = atoi(optarg);
      break;
    case 'l':
      *len = atoi(optarg);
      break;
    case 'g':
      *grid_dim = atoi(optarg);
      break;
    case 'b':
      *block_dim = atoi(optarg);
      break;
    default:break;
    }
  }
  if (*grid_dim == -1) {
    *grid_dim = (*num_rays + *block_dim-1) / *block_dim;
  }
}


int main(int argc, char* argv[]) {
  int num_rays;
  int len;
  int grid_dim;
  int block_dim;
  get_input(argc, argv, &num_rays, &len, &grid_dim, &block_dim);
  
  size_t size = len * len * sizeof(float);

  float* d_window;
  cudaMalloc((void **) &d_window, size);
  cudaMemset((void *) d_window, 0.0, size);
  
  cudaEvent_t start ,end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  
  cudaEventRecord(start, 0);
  ray_trace_kernel<<<grid_dim,block_dim>>>(d_window, len, num_rays);
  cudaEventRecord(end, 0);

  float time;
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&time, start, end);
  time = time / 1000;
  printf("ray_num\tgrid_dim\tblock_dim\ttime\n");
  printf("%d\t%d\t%d\t%f\n", num_rays, grid_dim, block_dim, time);

  float* window = (float*) malloc(size);
  cudaMemcpy(window, d_window, size, cudaMemcpyDeviceToHost);

  char filename[] = "ball.dat";
  FILE *f = fopen(filename, "wb");
  if (f != NULL) {
    fwrite(window, sizeof(float), len * len, f);
    fclose(f);
  } else {
    fprintf(stderr, "Error opening %s: ", filename);
    perror("");
    free(window);
    cuchk(cudaFree(d_window));
    exit(EXIT_FAILURE);
  }

  free(window);
  cuchk(cudaFree(d_window));
  return 0;
}
