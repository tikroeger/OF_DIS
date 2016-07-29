#ifndef __OPTICALFLOW_AUX_
#define __OPTICALFLOW_AUX_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

#include "image.h"

#if (SELECTCHANNEL==1 | SELECTCHANNEL==2)  // use single band image
/* warp a color image according to a flow. src is the input image, wx and wy, the input flow. dst is the warped image and mask contains 0 or 1 if the pixels goes outside/inside image boundaries */
void image_warp(image_t *dst, image_t *mask, const image_t *src, const image_t *wx, const image_t *wy);

/* compute image first and second order spatio-temporal derivatives of a color image */
void get_derivatives(const image_t *im1, const image_t *im2, const convolution_t *deriv, image_t *dx, image_t *dy, image_t *dt, image_t *dxx, image_t *dxy, image_t *dyy, image_t *dxt, image_t *dyt);

#else                                     // use RGB image_new
/* warp a color image according to a flow. src is the input image, wx and wy, the input flow. dst is the warped image and mask contains 0 or 1 if the pixels goes outside/inside image boundaries */
void image_warp(color_image_t *dst, image_t *mask, const color_image_t *src, const image_t *wx, const image_t *wy);

/* compute image first and second order spatio-temporal derivatives of a color image */
void get_derivatives(const color_image_t *im1, const color_image_t *im2, const convolution_t *deriv, color_image_t *dx, color_image_t *dy, color_image_t *dt, color_image_t *dxx, color_image_t *dxy, color_image_t *dyy, color_image_t *dxt, color_image_t *dyt);

#endif
    
    
/* compute the smoothness term */
void compute_smoothness(image_t *dst_horiz, image_t *dst_vert, const image_t *uu, const image_t *vv, const convolution_t *deriv_flow, const float quarter_alpha);
// void compute_smoothness_SF(image_t *dst_horiz, image_t *dst_vert, const image_t *xx1, const image_t *xx2, const image_t *yy, const image_t *xx3, const convolution_t *deriv_flow, const float quarter_alpha);

/* sub the laplacian (smoothness term) to the right-hand term */
void sub_laplacian(image_t *dst, const image_t *src, const image_t *weight_horiz, const image_t *weight_vert);

/* compute the dataterm and the matching term
   a11 a12 a22 represents the 2x2 diagonal matrix, b1 and b2 the right hand side
   other (color) images are input */
void compute_data_and_match(image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *mask, image_t *wx, image_t *wy, image_t *du, image_t *dv, image_t *uu, image_t *vv, color_image_t *Ix, color_image_t *Iy, color_image_t *Iz, color_image_t *Ixx, color_image_t *Ixy, color_image_t *Iyy, color_image_t *Ixz, color_image_t *Iyz, image_t *desc_weight, image_t *desc_flow_x, image_t *desc_flow_y, const float half_delta_over3, const float half_beta, const float half_gamma_over3);

/* compute the dataterm and ... REMOVED THE MATCHING TERM
   a11 a12 a22 represents the 2x2 diagonal matrix, b1 and b2 the right hand side
   other (color) images are input */
#if (SELECTCHANNEL==1 | SELECTCHANNEL==2)  // use single band image_delete
void compute_data(image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *mask, image_t *wx, image_t *wy, image_t *du, image_t *dv, image_t *uu, image_t *vv, image_t *Ix, image_t *Iy, image_t *Iz, image_t *Ixx, image_t *Ixy, image_t *Iyy, image_t *Ixz, image_t *Iyz, const float half_delta_over3, const float half_beta, const float half_gamma_over3);
#else
void compute_data(image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *mask, image_t *wx, image_t *wy, image_t *du, image_t *dv, image_t *uu, image_t *vv, color_image_t *Ix, color_image_t *Iy, color_image_t *Iz, color_image_t *Ixx, color_image_t *Ixy, color_image_t *Iyy, color_image_t *Ixz, color_image_t *Iyz, const float half_delta_over3, const float half_beta, const float half_gamma_over3);
#endif

#if (SELECTCHANNEL==1 | SELECTCHANNEL==2)  // use single band image_delete
void compute_data_DE(image_t *a11, image_t *b1, image_t *mask, image_t *wx, image_t *du, image_t *uu, image_t *Ix, image_t *Iy, image_t *Iz, image_t *Ixx, image_t *Ixy, image_t *Iyy, image_t *Ixz, image_t *Iyz, const float half_delta_over3, const float half_beta, const float half_gamma_over3);
#else
void compute_data_DE(image_t *a11, image_t *b1, image_t *mask, image_t *wx, image_t *du, image_t *uu, color_image_t *Ix, color_image_t *Iy, color_image_t *Iz, color_image_t *Ixx, color_image_t *Ixy, color_image_t *Iyy, color_image_t *Ixz, color_image_t *Iyz, const float half_delta_over3, const float half_beta, const float half_gamma_over3);
#endif


/* resize the descriptors to the new size using a weighted mean */
void descflow_resize(image_t *dst_flow_x, image_t *dst_flow_y, image_t *dst_weight, const image_t *src_flow_x, const image_t *src_flow_y, const image_t *src_weight);

/* resize the descriptors to the new size using a nearest neighbor method while keeping the descriptor with the higher weight at the end */
void descflow_resize_nn(image_t *dst_flow_x, image_t *dst_flow_y, image_t *dst_weight, const image_t *src_flow_x, const image_t *src_flow_y, const image_t *src_weight);

#ifdef __cplusplus
}
#endif


#endif
