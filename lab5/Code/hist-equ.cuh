#ifndef HIST_EQU_COLOR_H
#define HIST_EQU_COLOR_H

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    


PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
int* histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin);

__global__ void histogramGPU(int *hist_out, unsigned char * img_in, int nbr_bin);
__global__ void findLut(int d, int min, int* lut, int *hist_in);
__global__ void find_histogram_equalization(unsigned char * img_out, unsigned char * img_in, int *lut, int img_size);

//Contrast enhancement for gray-scale images
PGM_IMG contrast_enhancement_g(PGM_IMG img_in);	

#endif
