#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

PGM_IMG contrast_enhancement_g(PGM_IMG img_in, unsigned char *output)
{
    PGM_IMG result;
    result.w = img_in.w;
    result.h = img_in.h;

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    for (int i = 0; i < result.w*result.h; i++) {
        result.img[i] = output[i];
    }

  //  histogram(hist, img_in.img, img_in.h*img_in.w,256);

  //  histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);

  //  free(hist);
    
    return result;
}
