/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>

#define FILTER_RADIUS 3
#define FILTER_LENGTH 	(2 * FILTER_RADIUS + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.5 
#define BLOCK_WIDTH 64
#define TILE_WIDTH 67
#define CPU_CODE 10

__device__ __constant__ double d_filter[FILTER_LENGTH];
 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#ifdef CPU_CODE
////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(double *h_Dst, double *h_Src, double *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
  int padding = filterR;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + padding + k;

        sum += h_Src[(y+padding) * imageW + d] * h_Filter[filterR - k];
     
        h_Dst[(y+padding) * imageW + x + padding] = sum;
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
  int padding = filterR;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      double sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + padding + k;
        sum += h_Src[d * imageW + x + padding] * h_Filter[filterR - k];
 
        h_Dst[(y+padding)* imageW + x+padding] = sum;
      }
    }
  }
}
#endif

__global__ void convolution_kernel_x(double *d_output, double *d_input, int num_row, int num_col){

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

    result += input_array[ty * (TILE_WIDTH) + d] * d_filter[FILTER_RADIUS - k];
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

    result += input_array[d*TILE_WIDTH + tx + FILTER_RADIUS] * d_filter[FILTER_RADIUS-k];
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
  *h_outputGPU,
  *h_Input_block,
  *h_Buffer_block,
  *h_Output_block;

  double
  *d_OutputCPU,
  *d_Output_block,
  *d_Input_block,
  *d_Buffer_block;

  int imageW;
  int imageH;
  int block_size = BLOCK_WIDTH;
  unsigned int i;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;

  /*printf("Enter filter radius : ");
  scanf("%d", &filter_radius); */

  printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
  scanf("%d", &imageW);
  imageH = imageW;

  int padded_array_size = (2* FILTER_RADIUS + imageW) * (2 * FILTER_RADIUS + imageW);
  int padded_array_width = 2* FILTER_RADIUS + imageW;

  /*printf("Enter block size:  ");
  scanf("%d", &block_size);*/

  int padded_block_size = (block_size + 2 * FILTER_RADIUS) * (block_size + 2 * FILTER_RADIUS);
  int padded_block_width = block_size + 2 * FILTER_RADIUS;


  printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
  printf("Allocating and initializing host arrays...\n");

  h_Filter    = (double *)malloc(FILTER_LENGTH* sizeof(double));
  h_Input     = (double *)calloc(padded_array_size, sizeof(double));
  h_Buffer    = (double *)calloc(padded_array_size, sizeof(double));
  h_OutputCPU = (double *)calloc(padded_array_size, sizeof(double));
  h_outputGPU = (double *)calloc(padded_array_size, sizeof(double));
  h_Input_block = (double *)calloc(padded_block_size, sizeof(double));
  h_Output_block = (double *)calloc(padded_block_size, sizeof(double));
  h_Buffer_block = (double *)calloc(padded_block_size, sizeof(double));

  if(h_Filter == NULL || h_Input == NULL || h_Buffer == NULL || h_OutputCPU == NULL || h_outputGPU == NULL || h_Input_block == NULL|| h_Output_block == NULL || h_Buffer_block == NULL){
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
      }else{
        h_Input[i] = (double)rand() / ((double)RAND_MAX / 255) + (double)rand() / (double)RAND_MAX;
      }
    }
  }


  dim3 grid, block;
  if(block_size > 32){
      block.x = 32; 
      block.y = 32;
      grid.x = block_size/block.x;
      grid.y = block_size/block.y;
  }     
  else{
      block.x = block_size;
      block.y = block_size; 
      grid.x = 1;
      grid.y = 1; 
  } 

          //-------ALLOCATE DEVICE MEMORY-------//
  gpuErrchk(cudaMallocManaged((void**)&d_OutputCPU, padded_array_size * sizeof(double)));
  gpuErrchk(cudaMallocManaged((void**)&d_Buffer_block, padded_block_size * sizeof(double)));
  gpuErrchk(cudaMallocManaged((void**)&d_Input_block, padded_block_size * sizeof(double)));
  gpuErrchk(cudaMallocManaged((void**)&d_Output_block, padded_block_size * sizeof(double)));

  cudaMemcpyToSymbol(d_filter, h_Filter, sizeof(double) * FILTER_LENGTH);

  cudaEventRecord(start);

  for(int x=0;x<imageH/block_size;x++){
    for(int y=0;y<imageW/block_size;y++){

      for(int i = 0; i < padded_block_width; i++){
        for(int j = 0; j < padded_block_width; j++)
          h_Input_block[i*block_size+j] = h_Input[(x*block_size+i)*padded_array_width+y*block_size+j];
      }

      gpuErrchk(cudaMemcpy(d_Input_block, h_Input_block, sizeof(double) * padded_block_size, cudaMemcpyHostToDevice));

      convolution_kernel_x<<<grid, block>>>(d_Buffer_block, d_Input_block, block_size, block_size);

      gpuErrchk(cudaMemcpy(h_Buffer_block, d_Buffer_block, sizeof(double) * padded_block_size, cudaMemcpyDeviceToHost));

      for(int i = 0; i < padded_block_width; i++){
        for(int j = 0; j < padded_block_width; j++){
          h_Buffer[(x*block_size+i) * padded_array_width+y*block_size+j] = h_Buffer_block[i*block_size+j];
        }
      }
    }
  }
cudaDeviceSynchronize();

for(int x=0;x<imageH/block_size;x++){
  for(int y=0;y<imageW/block_size;y++){

    for(int i=0;i<padded_block_width;i++){
      for(int j=0;j<padded_block_width;j++){
        h_Buffer_block[j*block_size+i] = h_Buffer[(y*block_size+j)*padded_array_width+x*block_size+i];
      }
    }

    gpuErrchk(cudaMemcpy(d_Buffer_block, h_Buffer_block, sizeof(double)*padded_block_size, cudaMemcpyHostToDevice));

    convolution_kernel_y<<<grid, block>>>(d_Output_block, d_Buffer_block, block_size, block_size);

    gpuErrchk(cudaMemcpy(h_Output_block, d_Output_block, sizeof(double)*padded_block_size, cudaMemcpyDeviceToHost));

    for(int i=0;i<padded_block_width;i++){
      for(int j=0;j<padded_block_width;j++){
        h_outputGPU[(y*block_size+j)*padded_array_width+x*block_size+i] = h_Output_block[j*block_size+i];
      }
    }

  }
}
cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("\n\nGPU elapsed time: %f\n\n", milliseconds/1000);

#ifdef CPU_CODE
  int ap;
  clock_t start_CPU, end_CPU;
  float cpu_time_used;

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
        printf("k: %d\n", k);
        printf("ap is %d\n", ap);
        printf("OUT OF ACCURACY LEAVE\n");
        continue;
        //return(0);
      }
  }
#endif

  cudaFree(d_OutputCPU);
  cudaFree(d_Output_block);
  cudaFree(d_Input_block);
  cudaFree(d_Buffer_block);
  cudaFree(d_filter);

  // free all the allocated memory
  free(h_OutputCPU);
  free(h_Buffer);
  free(h_Input);
  free(h_Filter);
  free(h_Output_block);
  free(h_Input_block);
  free(h_Buffer_block);

  cudaDeviceReset();


    return 0;
}