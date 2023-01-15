#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

void run_cpu_gray_test(PGM_IMG img_in, char *out_filename, unsigned char *output);

int main(int argc, char *argv[]){
    PGM_IMG img_ibuf_g;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

	if (argc != 3) {
		printf("Run with input file name and output file name as arguments\n");
		exit(1);
	}

    unsigned char * input, *res;
    int *hist, *tmp, min=0, i=0, d;
    float *cdf;
    int img_size;
	
    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm(argv[1]);


    img_size = img_ibuf_g.w * img_ibuf_g.h;

    cudaEventRecord(start);
    cudaMallocManaged(&hist, 256*sizeof(int));
    cudaMallocManaged(&input,img_size * sizeof(unsigned char));
    cudaMallocManaged(&cdf, 256*sizeof(float));
    cudaMallocManaged(&tmp,256*sizeof(int));
    cudaMallocManaged(&res, img_size*sizeof(unsigned char));




    for (int i = 0; i < img_size; i++) {
        input[i] = img_ibuf_g.img[i];
        res[i] = 0;
    }

    for (int i = 0; i < 256; i++)
    {
        hist[i] = 0;
        tmp[i] = 0;
        cdf[i] = 0.0;
    }
    

    int THREADS = 512;
    int gridDim = (img_size + 1) / THREADS;

    d_histogram<<<gridDim, THREADS>>>(hist, input, img_size, 256);
    cudaDeviceSynchronize();
    d_cdf<<<1,256>>>(cdf,hist,img_size);
    cudaDeviceSynchronize();

    while(min==0){
        min = hist[i++];
    }
    
    d = img_size-min;
    d_lut<<<1, 256>>>(tmp, cdf, img_size, min, d);
    cudaDeviceSynchronize();

    d_create_out<<<gridDim, THREADS>>>(res, input, tmp, img_size);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %lf\n", milliseconds);


    run_cpu_gray_test(img_ibuf_g, argv[2], res);
    free_pgm(img_ibuf_g);

    cudaFree(hist);
    cudaFree(input);
    cudaFree(cdf);
    cudaFree(tmp);
    cudaFree(res);

	return 0;
}



void run_cpu_gray_test(PGM_IMG img_in, char *out_filename, unsigned char *output)
{
    PGM_IMG img_obuf;
    
    
    printf("Starting CPU processing...\n");
    img_obuf = contrast_enhancement_g(img_in, output);
    write_pgm(img_obuf, out_filename);
    free_pgm(img_obuf);
}


PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

        
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}

