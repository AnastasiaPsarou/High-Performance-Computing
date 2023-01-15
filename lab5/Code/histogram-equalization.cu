#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"


__global__ void d_histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    __shared__ int s_hist[256];
 
    int id = threadIdx.x+blockDim.x*blockIdx.x;
    int idx = threadIdx.x;
    if(idx < 256){
        s_hist[idx] = 0;
    }

    __syncthreads();
 
    for(int i=id; i<img_size; i+=blockDim.x*gridDim.x)
      atomicAdd(&s_hist[img_in[i]], 1);

    __syncthreads();

    if(idx < 256){
        atomicAdd(&hist_out[idx], s_hist[idx]);
    }
}

__global__ void d_cdf(float* cdf, int * hist_in, int img_size){
    __shared__ float s_arr[2*256];

    int id = threadIdx.x;
    if(id < 256)
        s_arr[id] = float(hist_in[id]);
        
    __syncthreads();

    for (unsigned int stride = 1; stride <= 256; stride *= 2) {
        __syncthreads();
		unsigned int index = (threadIdx.x + 1)*stride * 2 - 1;
		if(index < 256)
			s_arr[index] += s_arr[index - stride];
		
	}

	for (unsigned int stride = 256/2; stride>0; stride /= 2) {
		__syncthreads();
		unsigned int index = (threadIdx.x + 1)*stride * 2 - 1;
		if (index + stride < 256) {
			s_arr[index + stride] += s_arr[index];
		}
	}

	__syncthreads();
	if (id<256)
		cdf[id] += s_arr[threadIdx.x];
}

__global__ void d_lut(int * img_out, float * cdf, int img_size, int min, float d){

    int idx = threadIdx.x;
    int x;

    if(idx < 256){
        x =  (int)((cdf[idx]-min)*255/d + 0.5);
        if(x<0)
            x=0;
        img_out[idx] = x;
    }

}

__global__ void d_create_out(unsigned char * res, unsigned char * img_in, int * lut, int img_size){
    int id = threadIdx.x + blockDim.x*blockIdx.x;

    for(int i = id; i < img_size; i += blockDim.x*gridDim.x){
        if(lut[img_in[i]] > 255)    
            res[i] = (unsigned char)255;
        else
            res[i] = (unsigned char)(lut[img_in[i]]);
       
    }
}

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, min, d;
    int cdf;
    // /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
     i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0)
           lut[i] = 0;
    }
    
    /* Get the result image */
    for(i = 0; i < img_size; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
        
    }
}
