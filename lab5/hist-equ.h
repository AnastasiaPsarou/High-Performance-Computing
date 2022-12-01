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


__global__ void d_histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
__global__ void d_cdf(float* cdf, int * hist_in, int img_size);
__global__ void d_lut(int * img_out, float * cdf, int img_size, int min, float d);
__global__ void d_create_out(unsigned char * res, unsigned char * img_in, int *lut, int img_size);

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin);

//Contrast enhancement for gray-scale images
PGM_IMG contrast_enhancement_g(PGM_IMG img_in, unsigned char *output);

#endif
