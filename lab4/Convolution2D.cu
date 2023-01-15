/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/
#include <stdio.h>
#include <stdlib.h>

#define FILTER_RADIUS 2
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.05 
#define FILTER_LENGTH 	(2 * FILTER_RADIUS + 1)
#define BLOCK_WIDTH 32
#define TILE_WIDTH 34

__device__ __constant__ double d_Filter[FILTER_LENGTH];
 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + FILTER_RADIUS + k;

          sum += h_Src[(y+FILTER_RADIUS) * imageW + d] * h_Filter[filterR - k];  
        h_Dst[(y + FILTER_RADIUS) * imageW + x + FILTER_RADIUS] = sum;
      }
    }
  }
        
}
////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(double *h_Dst, double *h_Src, double *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + FILTER_RADIUS + k;

          sum += h_Src[d * imageW + x + FILTER_RADIUS] * h_Filter[filterR - k];
 
        h_Dst[(y + FILTER_RADIUS)* imageW + x + FILTER_RADIUS] = sum;
      }
    }
  }
}

__global__ void convolution_kernel_x(double *d_output, double *d_input,  int num_row, int num_col){
  
  int idx_x = blockIdx.x*blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y*blockDim.y + threadIdx.y;
  double result = 0.f;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int k;
  __shared__ double input_array[TILE_WIDTH * TILE_WIDTH];

  input_array[ty * TILE_WIDTH + tx + FILTER_RADIUS] = d_input[idx_y *(num_row+2*FILTER_RADIUS) + idx_x + FILTER_RADIUS];
  
  //thesi gia to de3ia padding
  if(tx >= (BLOCK_WIDTH -  FILTER_RADIUS)){
    input_array[ty * TILE_WIDTH + FILTER_RADIUS + 2 * BLOCK_WIDTH - tx - 1] = 0;
  }
  //thesi gia to aristero padding
  if(tx <= (FILTER_RADIUS - 1)){
    input_array[ty * TILE_WIDTH + FILTER_RADIUS - tx - 1] = 0;
  }
  __syncthreads();
  
  for(k = -FILTER_RADIUS; k <= FILTER_RADIUS; k++){
    int d = tx + FILTER_RADIUS + k;

    result += input_array[ty * (TILE_WIDTH) + d] * d_Filter[FILTER_RADIUS - k];
  }
  __syncthreads();

  d_output[(idx_y + FILTER_RADIUS) * num_row + idx_x + FILTER_RADIUS] = result;
}


__global__ void convolution_kernel_y(double *d_output, double *d_input, int num_row, int num_col){

  int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
  int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  double result = 0.f;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int k;
  __shared__ double input_array[TILE_WIDTH * TILE_WIDTH];

  input_array[tx * TILE_WIDTH + ty + FILTER_RADIUS] = d_input[idx_x * (num_row + 2*FILTER_RADIUS) + idx_y + FILTER_RADIUS];
  
  //thesi gia to katw padding
  if(ty >= (BLOCK_WIDTH -  FILTER_RADIUS)){
    input_array[tx * TILE_WIDTH + FILTER_RADIUS + 2 * BLOCK_WIDTH - ty - 1] = 0;
  }
  //thesi gia to panw padding
  if(ty <= (FILTER_RADIUS - 1)){
    input_array[tx * TILE_WIDTH - ty - 1 + FILTER_RADIUS] = 0;
  }
  __syncthreads();

  for(k = -FILTER_RADIUS; k <= FILTER_RADIUS; k++){
    int d = ty + FILTER_RADIUS + k;

    result += input_array[d*TILE_WIDTH + tx + FILTER_RADIUS] * d_Filter[FILTER_RADIUS-k];
  }
  __syncthreads();
   
  d_output[(idx_y + FILTER_RADIUS) * num_row + idx_x + FILTER_RADIUS] = result;
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
  double
  *h_Filter,
  *h_Input,
  *h_Buffer,
  *h_OutputCPU,
  *h_outputGPU;

  double
  *d_Input,
  *d_Buffer,
  *d_OutputCPU;

  int imageW;
  int imageH;
  unsigned int i;
  double ap;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
  clock_t start_CPU, end_CPU;
  float cpu_time_used;

  /*printf("Enter filter radius : ");
  scanf("%d", &filter_radius);*/

  printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
  scanf("%d", &imageW);
  imageH = imageW;

  int padded_array_size = (2* FILTER_RADIUS + imageW) * (2 * FILTER_RADIUS + imageW);
  int padded_array_width = 2* FILTER_RADIUS + imageW;
  int shared_memory_size = (2* FILTER_RADIUS + BLOCK_WIDTH) * (2 * FILTER_RADIUS + BLOCK_WIDTH) * sizeof(double);

  printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
  printf("Allocating and initializing host arrays...\n");
  
  h_Filter    = (double *)malloc(FILTER_LENGTH * sizeof(double));
  h_Input     = (double *)calloc(padded_array_size, sizeof(double));
  h_Buffer    = (double *)calloc(padded_array_size, sizeof(double));
  h_OutputCPU = (double *)calloc(padded_array_size, sizeof(double));
  h_outputGPU = (double *)calloc(padded_array_size, sizeof(double));

  if(h_Filter == NULL || h_Input == NULL || h_Buffer == NULL || h_OutputCPU == NULL || h_outputGPU == NULL){
    fprintf(stderr,"calloc failure: %d\n", __LINE__);
    if (abort) exit(1);
  }

  srand(200);

  for (i = 0; i < FILTER_LENGTH; i++) {
      h_Filter[i] = (double)(rand() % 16);
  }

  for (int i = 0; i < padded_array_width; i++) {
    for (int j = 0; j < padded_array_width; j++) {
      if (i < FILTER_RADIUS || i > imageW + FILTER_RADIUS - 1 || j < FILTER_RADIUS || j > FILTER_RADIUS + imageW - 1) {
          //init padding with 0
      }
      else{
        h_Input[i] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
      }
    }
  }

  dim3 grid, block;
  if(imageH > 32){
      block.x = 32; 
      block.y = 32;
      int grid_size = (imageH *imageW)/1024;
      grid.x = sqrt(grid_size);
      grid.y = sqrt(grid_size); 
  }     
  else{
      block.x = imageH;
      block.y = imageH; 
      grid.x = 1;
      grid.y = 1; 
  } 

  cudaMallocManaged((void**)&d_Input, padded_array_size * sizeof(double));
  cudaMallocManaged((void**)&d_Buffer, padded_array_size * sizeof(double));
  cudaMallocManaged((void**)&d_OutputCPU, padded_array_size * sizeof(double));

  cudaMemcpyToSymbol(d_Filter, h_Filter, sizeof(double) * FILTER_LENGTH);
  
  gpuErrchk(cudaMemcpy(d_Input, h_Input, padded_array_size * sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_Buffer, h_Buffer, padded_array_size * sizeof(double), cudaMemcpyHostToDevice));
  
  printf("GPU computation...\n");

  cudaEventRecord(start);
  convolution_kernel_x<<<grid, block, shared_memory_size>>>(d_Buffer, d_Input, imageH, imageH);
  cudaDeviceSynchronize();

  convolution_kernel_y<<<grid, block, shared_memory_size>>>(d_OutputCPU, d_Buffer, imageH, imageH);    
  cudaDeviceSynchronize();

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("\n\nGPU elapsed time: %f\n\n", milliseconds);

  // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
  printf("CPU computation...\n");

  start_CPU = clock();

  convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, FILTER_RADIUS); // convolution kata grammes
  convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, FILTER_RADIUS); // convolution kata sthles

  end_CPU = clock();
  cpu_time_used = ((float) (end_CPU - start_CPU)) / CLOCKS_PER_SEC;
  printf("\n\nCPU elapsed time: %f\n\n", cpu_time_used);

  for(int k = 0; k < padded_array_size; k++){
    ap = abs(h_outputGPU[k] - h_OutputCPU[k]);
      if(ap > accuracy){
        printf("ap is %lf\n", ap);
        printf("k is %d\n", k);
        printf("GPU: %lf vs CPU: %lf", h_outputGPU[k], h_OutputCPU[k]);
        printf("OUT OF ACCURACY LEAVE\n");
        return(0);
      }
  }

  cudaFree(d_Input);
  cudaFree(d_Buffer);
  cudaFree(d_OutputCPU);
  cudaFree(d_Filter);

  free(h_OutputCPU);
  free(h_Buffer);
  free(h_Input);
  free(h_Filter);

  cudaDeviceReset();

  return 0;
}