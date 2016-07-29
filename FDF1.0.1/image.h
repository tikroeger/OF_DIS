#ifndef __IMAGE_H_
#define __IMAGE_H_

#include <stdio.h>

#define MIN_TA(a, b) ((a) < (b) ? (a) : (b))
#define MAX_TA(a, b) ((a) > (b) ? (a) : (b))
#define MINMAX_TA(a,b) MIN_TA( MAX_TA(a,0) , b-1 )

#ifdef __cplusplus
extern "C" {
#endif


/********** STRUCTURES *********/

/* structure for 1-channel image */
typedef struct image_s
{
  int width;		/* Width of the image */
  int height;		/* Height of the image */
  int stride;		/* Width of the memory (width + paddind such that it is a multiple of 4) */
  float *c1;		/* Image data, aligned */
} image_t;

/* structure for 3-channels image stored with one layer per color, it assumes that c2 = c1+width*height and c3 = c2+width*height. */
typedef struct color_image_s
{
    int width;			/* Width of the image */
    int height;			/* Height of the image */
    int stride;         /* Width of the memory (width + paddind such that it is a multiple of 4) */
    float *c1;			/* Color 1, aligned */
    float *c2;			/* Color 2, consecutive to c1*/
    float *c3;			/* Color 3, consecutive to c2 */
} color_image_t;

/* structure for color image pyramid */
typedef struct color_image_pyramid_s 
{
  float scale_factor;          /* difference of scale between two levels */
  int min_size;                /* minimum size for width or height at the coarsest level */
  int size;                    /* number of levels in the pyramid */
  color_image_t **images;      /* list of images with images[0] the original one, images[size-1] the finest one */
} color_image_pyramid_t;

/* structure for convolutions */
typedef struct convolution_s
{
    int order;			/* Order of the convolution */
    float *coeffs;		/* Coefficients */
    float *coeffs_accu;	/* Accumulated coefficients */
} convolution_t;

/********** Create/Delete **********/

/* allocate a new image of size width x height */
image_t *image_new(const int width, const int height);

/* allocate a new image and copy the content from src */
image_t *image_cpy(const image_t *src);

/* set all pixels values to zeros */
void image_erase(image_t *image);

/* free memory of an image */
void image_delete(image_t *image);

/* multiply an image by a scalar */
void image_mul_scalar(image_t *image, const float scalar);

/* allocate a new color image of size width x height */
color_image_t *color_image_new(const int width, const int height);

/* allocate a new color image and copy the content from src */
color_image_t *color_image_cpy(const color_image_t *src);

/* set all pixels values to zeros */
void color_image_erase(color_image_t *image);

/* free memory of a color image */
void color_image_delete(color_image_t *image);

/* reallocate the memory of an image to fit the new width height */
void resize_if_needed_newsize(image_t *im, const int w, const int h);

/************ Resizing *********/

/* resize an image with bilinear interpolation */
image_t *image_resize_bilinear(const image_t *src, const float scale);

/* resize an image with bilinear interpolation to fit the new weidht, height ; reallocation is done if necessary */
void image_resize_bilinear_newsize(image_t *dst, const image_t *src, const int new_width, const int new_height);

/* resize a color image  with bilinear interpolation */
color_image_t *color_image_resize_bilinear(const color_image_t *src, const float scale);

/************ Convolution ******/

/* return half coefficient of a gaussian filter */
float *gaussian_filter(const float sigma, int *fSize);

/* create a convolution structure with a given order, half_coeffs, symmetric or anti-symmetric according to even parameter */
convolution_t *convolution_new(int order, const float *half_coeffs, const int even);

/* perform an horizontal convolution of an image */
void convolve_horiz(image_t *dest, const image_t *src, const convolution_t *conv);

/* perform a vertical convolution of an image */
void convolve_vert(image_t *dest, const image_t *src, const convolution_t *conv);

/* free memory of a convolution structure */
void convolution_delete(convolution_t *conv);

/* perform horizontal and/or vertical convolution to a color image */
void color_image_convolve_hv(color_image_t *dst, const color_image_t *src, const convolution_t *horiz_conv, const convolution_t *vert_conv);

/* perform horizontal and/or vertical convolution to a single band image */
void image_convolve_hv(image_t *dst, const image_t *src, const convolution_t *horiz_conv, const convolution_t *vert_conv);


/************ Pyramid **********/

/* create a pyramid of color images using a given scale factor, stopping when one dimension reach min_size and with applying a gaussian smoothing of standard deviation spyr (no smoothing if 0) */
color_image_pyramid_t *color_image_pyramid_create(const color_image_t *src, const float scale_factor, const int min_size, const float spyr);

/* delete the structure of a pyramid of color images */
void color_image_pyramid_delete(color_image_pyramid_t *pyr);

#ifdef __cplusplus
}
#endif


#endif


