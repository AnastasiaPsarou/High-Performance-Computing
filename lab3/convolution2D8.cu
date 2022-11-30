/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.5 
 
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
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
  int filterL = (2*filterR+1);
  int padding = floor(filterL/2);
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

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
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  int filterL = (2*filterR+1);
  int padding = floor(filterL/2);
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + padding + k;
        sum += h_Src[d * imageW + x + padding] * h_Filter[filterR - k];
 
        h_Dst[(y+padding)* imageW + x+padding] = sum;
      }
    }
  }
}

__global__ void convolution_kernel_x(float *d_output, float *d_input, float *d_filter, int num_row, int num_col, int filterR){

  int idx_x = blockIdx.x*blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y*blockDim.y + threadIdx.y;
  float result = 0.f;
  int filterL = (2*filterR+1);
  int padding = floor(filterL/2);

  for(int k = -filterR; k <= filterR; k++){
    int d = idx_x + padding + k;

    result += d_input[(idx_y+padding)*num_col+d] * d_filter[filterR-k];
  }
  d_output[(idx_y+padding)*num_row+idx_x+padding] = result;
}


__global__ void convolution_kernel_y(float *d_output, float *d_input, float *d_filter, int num_row, int num_col, int filterR){
  int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
  int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  float result = 0.f;
  int filterL = (2*filterR+1);
  int padding = floor(filterL/2);

  for(int k = -filterR; k<=filterR; k++){
    int d = idx_x + padding + k;

    result += d_input[num_row*d+idx_y+padding] * d_filter[filterR-k];
  }
   
  d_output[(idx_x+padding)*num_row+idx_y+padding] = result;
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    float
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_outputGPU;

    float
    *d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputCPU;

    int imageW;
    int imageH;
    int padding;
    unsigned int i;
    float ap;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    clock_t start_CPU, end_CPU;
    float cpu_time_used;

	  printf("Enter filter radius : ");
	  scanf("%d", &filter_radius);

    //imageW and imageH are given by the user and we consider them to be equal
    //imageW = imageH = N, where N is given by the user
    //also we consider that the images are squared  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

    padding = floor(FILTER_LENGTH/2);

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");

    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    h_Input     = (float *)malloc((imageW+2*padding) * (imageH+2*padding) * sizeof(float));
    h_Buffer    = (float *)malloc((imageW+2*padding) * (imageH+2*padding) * sizeof(float));
    h_OutputCPU = (float *)malloc((imageW+2*padding) * (imageH+2*padding) * sizeof(float));
    h_outputGPU = (float *)malloc((imageW+2*padding) * (imageH+2*padding)*sizeof(float));
    if(h_Filter == NULL || h_Input == NULL || h_Buffer == NULL || h_OutputCPU == NULL || h_outputGPU == NULL){
      fprintf(stderr,"Malloc failure: %d\n", __LINE__);
      if (abort) exit(1);
    }

    gpuErrchk(cudaMalloc((void**)&d_Filter, FILTER_LENGTH*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_Input, (imageW+2*padding)*(imageH+2*padding)*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_Buffer, (imageW+2*padding)*(imageH+2*padding)*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_OutputCPU, ((imageW+2*padding) * (imageH+2*padding)*sizeof(float))));


    // h_Filter is the filter that is used for the convolution and it's randomly initialised
    // h_Input is the image in which the convolution is taking place and is also randomly initialised

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }

    for (i = 0; i < (imageW+2*padding) * (imageH+2*padding); i++) {
        h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
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

    cudaEventRecord(start);
    gpuErrchk(cudaMemcpy(d_Filter, h_Filter, sizeof(float) * FILTER_LENGTH, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_Input, h_Input, (imageW+2*padding) * (imageH+2*padding) * sizeof(float), cudaMemcpyHostToDevice));


    convolution_kernel_x<<<grid, block>>>(d_Buffer, d_Input, d_Filter, imageH, imageH, filter_radius);
    cudaDeviceSynchronize();

    convolution_kernel_y<<<grid, block>>>(d_OutputCPU, d_Buffer, d_Filter, imageH, imageH, filter_radius);    
    cudaDeviceSynchronize();

    gpuErrchk(cudaMemcpy(h_outputGPU, d_OutputCPU, sizeof(float)*(imageW+2*padding) * (imageH+2*padding), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\n\nGPU elapsed time: %f\n\n", milliseconds/1000);

    // this is the part that is executed on the CPU and based on that the comparison with the GPU should take place
    printf("CPU computation...\n");


    start_CPU = clock();
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution by lines
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution by columns 
    end_CPU = clock();
    cpu_time_used = ((float) (end_CPU - start_CPU)) / CLOCKS_PER_SEC;
    printf("\n\nCPU elapsed time: %f\n\n", cpu_time_used);

    for(int k = 0; k < (imageW+2*padding) * (imageH+2*padding); k++){
      ap = abs(h_outputGPU[k] - h_OutputCPU[k]);
        if(ap > accuracy){
          printf("OUT OF ACCURACY LEAVE\n");
          return(0);
        }
    }


    // The comparison between CPU and GPU takes place and if at least on result has bigger accuracy than the one we have set
    // then there is error and we can terminate the program  

    cudaFree(d_Input);
    cudaFree(d_Buffer);
    cudaFree(d_OutputCPU);
    cudaFree(d_Filter);


    // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);

    cudaDeviceReset();


    return 0;
}