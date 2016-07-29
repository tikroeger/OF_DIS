#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>

#include "image.h"

#include <xmmintrin.h>
typedef __v4sf v4sf;

/********** Create/Delete **********/

/* allocate a new image of size width x height */
image_t *image_new(const int width, const int height){
    image_t *image = (image_t*) malloc(sizeof(image_t));
    if(image == NULL){
        fprintf(stderr, "Error: image_new() - not enough memory !\n");
        exit(1);
    }
    image->width = width;
    image->height = height;  
    image->stride = ( (width+3) / 4 ) * 4;
    image->c1 = (float*) memalign(16, image->stride*height*sizeof(float));
    if(image->c1== NULL){
        fprintf(stderr, "Error: image_new() - not enough memory !\n");
        exit(1);
    }
    return image;
}

/* allocate a new image and copy the content from src */
image_t *image_cpy(const image_t *src){
    image_t *dst = image_new(src->width, src->height);
    memcpy(dst->c1, src->c1, src->stride*src->height*sizeof(float));
    return dst;
}

/* set all pixels values to zeros */
void image_erase(image_t *image){
    memset(image->c1, 0, image->stride*image->height*sizeof(float));
}


/* multiply an image by a scalar */
void image_mul_scalar(image_t *image, const float scalar){
    int i;
    v4sf* imp = (v4sf*) image->c1;
    const v4sf scalarp = {scalar,scalar,scalar,scalar};
    for( i=0 ; i<image->stride/4*image->height ; i++){
        (*imp) *= scalarp;
        imp+=1;
    }
}

/* free memory of an image */
void image_delete(image_t *image){
    if(image == NULL){
        //fprintf(stderr, "Warning: Delete image --> Ignore action (image not allocated)\n");
    }else{
    free(image->c1);
    free(image);
    }
}


/* allocate a new color image of size width x height */
color_image_t *color_image_new(const int width, const int height){
    color_image_t *image = (color_image_t*) malloc(sizeof(color_image_t));
    if(image == NULL){
        fprintf(stderr, "Error: color_image_new() - not enough memory !\n");
        exit(1);
    }
    image->width = width;
    image->height = height;  
    image->stride = ( (width+3) / 4 ) * 4;
    image->c1 = (float*) memalign(16, 3*image->stride*height*sizeof(float));
    if(image->c1 == NULL){
        fprintf(stderr, "Error: color_image_new() - not enough memory !\n");
        exit(1);
    }
    image->c2 =  image->c1+image->stride*height;
    image->c3 =  image->c2+image->stride*height;
    return image;
}

/* allocate a new color image and copy the content from src */
color_image_t *color_image_cpy(const color_image_t *src){
    color_image_t *dst = color_image_new(src->width, src->height);
    memcpy(dst->c1, src->c1, 3*src->stride*src->height*sizeof(float));
    return dst;
}

/* set all pixels values to zeros */
void color_image_erase(color_image_t *image){
    memset(image->c1, 0, 3*image->stride*image->height*sizeof(float));
}

/* free memory of a color image */
void color_image_delete(color_image_t *image){
    if(image){
        free(image->c1); // c2 and c3 was allocated at the same moment
        free(image);
    }
}

/* reallocate the memory of an image to fit the new width height */
void resize_if_needed_newsize(image_t *im, const int w, const int h){
    if(im->width != w || im->height != h){
        im->width = w;
        im->height = h;
        im->stride = ((w+3)/4)*4;
        float *data = (float *) memalign(16, im->stride*h*sizeof(float));
        if(data == NULL){
            fprintf(stderr, "Error: resize_if_needed_newsize() - not enough memory !\n");
            exit(1);
	    }
        free(im->c1);
        im->c1 = data;
    }
}


/************ Resizing *********/

/* resize an image to a new size (assumes a difference only in width) */
static void image_resize_horiz(image_t *dst, const image_t *src){
    const float real_scale = ((float) src->width-1) / ((float) dst->width-1);
    int i;
    for(i = 0; i < dst->height; i++){
        int j;
        for(j = 0; j < dst->width; j++){
            const int x = floor((float) j * real_scale);
	        const float dx = j * real_scale - x; 
	        if(x >= (src->width - 1)){
	            dst->c1[i * dst->stride + j] = src->c1[i * src->stride + src->width - 1]; 
            }else{
	            dst->c1[i * dst->stride + j] = 
		            (1.0f - dx) * src->c1[i * src->stride + x    ] + 
		            (       dx) * src->c1[i * src->stride + x + 1];
            }
        }
    }
}

/* resize a color image to a new size (assumes a difference only in width) */
static void color_image_resize_horiz(color_image_t *dst, const color_image_t *src){
    const float real_scale = ((float) src->width-1) / ((float) dst->width-1);
    int i;
    for(i = 0; i < dst->height; i++){
        int j;
        for(j = 0; j < dst->width; j++){
            const int x = floor((float) j * real_scale);
	        const float dx = j * real_scale - x; 
	        if(x >= (src->width - 1)){
	            dst->c1[i * dst->stride + j] = src->c1[i * src->stride + src->width - 1]; 
	            dst->c2[i * dst->stride + j] = src->c2[i * src->stride + src->width - 1]; 
	            dst->c3[i * dst->stride + j] = src->c3[i * src->stride + src->width - 1]; 
            }else{
	            dst->c1[i * dst->stride + j] = 
		            (1.0f - dx) * src->c1[i * src->stride + x    ] + 
		            (       dx) * src->c1[i * src->stride + x + 1];
	            dst->c2[i * dst->stride + j] = 
		            (1.0f - dx) * src->c2[i * src->stride + x    ] + 
		            (       dx) * src->c2[i * src->stride + x + 1];
	            dst->c3[i * dst->stride + j] = 
		            (1.0f - dx) * src->c3[i * src->stride + x    ] + 
		            (       dx) * src->c3[i * src->stride + x + 1];
            }
        }
    }
}

/* resize an image to a new size (assumes a difference only in height) */
static void image_resize_vert(image_t *dst, const image_t *src){
    const float real_scale = ((float) src->height-1) / ((float) dst->height-1);
    int i;
    for(i = 0; i < dst->width; i++){
        int j;
        for(j = 0; j < dst->height; j++){
	    const int y = floor((float) j * real_scale);
        const float dy = j * real_scale - y;
	    if(y >= (src->height - 1)){
	            dst->c1[j * dst->stride + i] = src->c1[i + (src->height - 1) * src->stride]; 
            }else{
	            dst->c1[j * dst->stride + i] =
		            (1.0f - dy) * src->c1[i + (y    ) * src->stride] + 
		            (       dy) * src->c1[i + (y + 1) * src->stride];
            }
        }
    }
}

/* resize a color image to a new size (assumes a difference only in height) */
static void color_image_resize_vert(color_image_t *dst, const color_image_t *src){
    const float real_scale = ((float) src->height) / ((float) dst->height);
    int i;
    for(i = 0; i < dst->width; i++){
        int j;
        for(j = 0; j < dst->height; j++){
	    const int y = floor((float) j * real_scale);
        const float dy = j * real_scale - y;
	    if(y >= (src->height - 1)){
            dst->c1[j * dst->stride + i] = src->c1[i + (src->height - 1) * src->stride]; 
	        dst->c2[j * dst->stride + i] = src->c2[i + (src->height - 1) * src->stride]; 
	        dst->c3[j * dst->stride + i] = src->c3[i + (src->height - 1) * src->stride]; 
        }else{
	        dst->c1[j * dst->stride + i] =
		        (1.0f - dy) * src->c1[i +  y      * src->stride] + 
		        (       dy) * src->c1[i + (y + 1) * src->stride];
            dst->c2[j * dst->stride + i] =
		        (1.0f - dy) * src->c2[i +  y      * src->stride] + 
		        (       dy) * src->c2[i + (y + 1) * src->stride];
		    dst->c3[j * dst->stride + i] =
		        (1.0f - dy) * src->c3[i +  y      * src->stride] + 
		        (       dy) * src->c3[i + (y + 1) * src->stride];
            }
        }
    }
}

/* return a resize version of the image with bilinear interpolation */
image_t *image_resize_bilinear(const image_t *src, const float scale){
    const int width = src->width, height = src->height;
    const int newwidth = (int) (1.5f + (width-1) / scale); // 0.5f for rounding instead of flooring, and the remaining comes from scale = (dst-1)/(src-1)
    const int newheight = (int) (1.5f + (height-1) / scale);
    image_t *dst = image_new(newwidth,newheight);
    if(height*newwidth < width*newheight){
        image_t *tmp = image_new(newwidth,height);
        image_resize_horiz(tmp,src);
        image_resize_vert(dst,tmp);
        image_delete(tmp);
    }else{
        image_t *tmp = image_new(width,newheight);
        image_resize_vert(tmp,src);
        image_resize_horiz(dst,tmp);
        image_delete(tmp);
    }
    return dst;
}

/* resize an image with bilinear interpolation to fit the new weidht, height ; reallocation is done if necessary */
void image_resize_bilinear_newsize(image_t *dst, const image_t *src, const int new_width, const int new_height){
    resize_if_needed_newsize(dst,new_width,new_height);
    if(new_width < new_height){
        image_t *tmp = image_new(new_width,src->height);
        image_resize_horiz(tmp,src);
        image_resize_vert(dst,tmp);
        image_delete(tmp);
    }else{
        image_t *tmp = image_new(src->width,new_height);
        image_resize_vert(tmp,src);
        image_resize_horiz(dst,tmp); 
        image_delete(tmp);
    }
}

/* resize a color image  with bilinear interpolation */
color_image_t *color_image_resize_bilinear(const color_image_t *src, const float scale){
    const int width = src->width, height = src->height;
    const int newwidth = (int) (1.5f + (width-1) / scale); // 0.5f for rounding instead of flooring, and the remaining comes from scale = (dst-1)/(src-1)
    const int newheight = (int) (1.5f + (height-1) / scale);
    color_image_t *dst = color_image_new(newwidth,newheight);
    if(height*newwidth < width*newheight){
        color_image_t *tmp = color_image_new(newwidth,height);
        color_image_resize_horiz(tmp,src);
        color_image_resize_vert(dst,tmp);
        color_image_delete(tmp);
    }else{
        color_image_t *tmp = color_image_new(width,newheight);
        color_image_resize_vert(tmp,src);
        color_image_resize_horiz(dst,tmp);
        color_image_delete(tmp);
    }
    return dst;
}

/************ Convolution ******/

/* return half coefficient of a gaussian filter
Details:
- return a float* containing the coefficient from middle to border of the filter, so starting by 0,
- it so contains half of the coefficient.
- sigma is the standard deviation.
- filter_order is an output where the size of the output array is stored */
float *gaussian_filter(const float sigma, int *filter_order){
    if(sigma == 0.0f){
        fprintf(stderr, "gaussian_filter() error: sigma is zeros\n");
        exit(1);
    }
    if(!filter_order){
        fprintf(stderr, "gaussian_filter() error: filter_order is null\n");
        exit(1);
    }
    // computer the filter order as 1 + 2* floor(3*sigma)
    *filter_order = floor(3*sigma); 
    if ( *filter_order == 0 )
        *filter_order = 1; 
    // compute coefficients
    float *data = (float*) malloc(sizeof(float) * (2*(*filter_order)+1));
    if(data == NULL ){
        fprintf(stderr, "gaussian_filter() error: not enough memory\n");
        exit(1);
    }
    const float alpha = 1.0f/(2.0f*sigma*sigma);
    float sum = 0.0f;
    int i;
    for(i=-(*filter_order) ; i<=*filter_order ; i++){
        data[i+(*filter_order)] = exp(-i*i*alpha);
        sum += data[i+(*filter_order)];
    }
    for(i=-(*filter_order) ; i<=*filter_order ; i++){
        data[i+(*filter_order)] /= sum;
    }
    // fill the output
    float *data2 = (float*) malloc(sizeof(float)*(*filter_order+1));
    if(data2 == NULL ){
        fprintf(stderr, "gaussian_filter() error: not enough memory\n");
        exit(1);
    }
    memcpy(data2, &data[*filter_order], sizeof(float)*(*filter_order)+sizeof(float));
    free(data);
    return data2;
}

/* given half of the coef, compute the full coefficients and the accumulated coefficients */
static void convolve_extract_coeffs(const int order, const float *half_coeffs, float *coeffs, float *coeffs_accu, const int even){
    int i;
    float accu = 0.0;
    if(even){
        for(i = 0 ; i <= order; i++){
	        coeffs[order - i] = coeffs[order + i] = half_coeffs[i];
        }
        for(i = 0 ; i <= order; i++){
	        accu += coeffs[i];
	        coeffs_accu[2 * order - i] = coeffs_accu[i] = accu;
        }
    }else{
        for(i = 0; i <= order; i++){
	        coeffs[order - i] = +half_coeffs[i];
	        coeffs[order + i] = -half_coeffs[i];
        }
        for(i = 0 ; i <= order; i++){
            accu += coeffs[i];
	        coeffs_accu[i] = accu;
	        coeffs_accu[2 * order - i]= -accu;
        }
    }
}

/* create a convolution structure with a given order, half_coeffs, symmetric or anti-symmetric according to even parameter */
convolution_t *convolution_new(const int order, const float *half_coeffs, const int even){
    convolution_t *conv = (convolution_t *) malloc(sizeof(convolution_t));
    if(conv == NULL){
        fprintf(stderr, "Error: convolution_new() - not enough memory !\n");
        exit(1);
    }
    conv->order = order;
    conv->coeffs = (float *) malloc((2 * order + 1) * sizeof(float));
    if(conv->coeffs == NULL){
        fprintf(stderr, "Error: convolution_new() - not enough memory !\n");
        free(conv);
        exit(1);
    }
    conv->coeffs_accu = (float *) malloc((2 * order + 1) * sizeof(float));
    if(conv->coeffs_accu == NULL){
        fprintf(stderr, "Error: convolution_new() - not enough memory !\n");
        free(conv->coeffs);
        free(conv);
        exit(1);
    }
    convolve_extract_coeffs(order, half_coeffs, conv->coeffs,conv->coeffs_accu, even);
    return conv;
}

static void convolve_vert_fast_3(image_t *dst, const image_t *src, const convolution_t *conv){
    const int iterline = (src->stride>>2)+1;
    const float *coeff = conv->coeffs;
    //const float *coeff_accu = conv->coeffs_accu;
    v4sf *srcp = (v4sf*) src->c1, *dstp = (v4sf*) dst->c1;
    v4sf *srcp_p1 = (v4sf*) (src->c1+src->stride);
    int i;
    for(i=iterline ; --i ; ){ // first line
        *dstp = (coeff[0]+coeff[1])*(*srcp) + coeff[2]*(*srcp_p1);
        dstp+=1; srcp+=1; srcp_p1+=1;
    }
    v4sf* srcp_m1 = (v4sf*) src->c1; 
    for(i=src->height-1 ; --i ; ){ // others line
        int j;
        for(j=iterline ; --j ; ){
            *dstp = coeff[0]*(*srcp_m1) + coeff[1]*(*srcp) + coeff[2]*(*srcp_p1);
            dstp+=1; srcp_m1+=1; srcp+=1; srcp_p1+=1;
        }
    }       
    for(i=iterline ; --i ; ){ // last line
        *dstp = coeff[0]*(*srcp_m1) + (coeff[1]+coeff[2])*(*srcp);
        dstp+=1; srcp_m1+=1; srcp+=1; 
    }  
}

static void convolve_vert_fast_5(image_t *dst, const image_t *src, const convolution_t *conv){
    const int iterline = (src->stride>>2)+1;
    const float *coeff = conv->coeffs;
    //const float *coeff_accu = conv->coeffs_accu;
    v4sf *srcp = (v4sf*) src->c1, *dstp = (v4sf*) dst->c1;
    v4sf *srcp_p1 = (v4sf*) (src->c1+src->stride);
    v4sf *srcp_p2 = (v4sf*) (src->c1+2*src->stride);
    int i;
    for(i=iterline ; --i ; ){ // first line
        *dstp = (coeff[0]+coeff[1]+coeff[2])*(*srcp) + coeff[3]*(*srcp_p1) + coeff[4]*(*srcp_p2);
        dstp+=1; srcp+=1; srcp_p1+=1; srcp_p2+=1;
    }
    v4sf* srcp_m1 = (v4sf*) src->c1;
    for(i=iterline ; --i ; ){ // second line
        *dstp = (coeff[0]+coeff[1])*(*srcp_m1) + coeff[2]*(*srcp) + coeff[3]*(*srcp_p1) + coeff[4]*(*srcp_p2);
        dstp+=1; srcp_m1+=1; srcp+=1; srcp_p1+=1; srcp_p2+=1;
    }   
    v4sf* srcp_m2 = (v4sf*) src->c1;
    for(i=src->height-3 ; --i ; ){ // others line
        int j;
        for(j=iterline ; --j ; ){
            *dstp = coeff[0]*(*srcp_m2) + coeff[1]*(*srcp_m1) + coeff[2]*(*srcp) + coeff[3]*(*srcp_p1) + coeff[4]*(*srcp_p2);
            dstp+=1; srcp_m2+=1;srcp_m1+=1; srcp+=1; srcp_p1+=1; srcp_p2+=1;
        }
    }    
    for(i=iterline ; --i ; ){ // second to last line
        *dstp = coeff[0]*(*srcp_m2) + coeff[1]*(*srcp_m1) + coeff[2]*(*srcp) + (coeff[3]+coeff[4])*(*srcp_p1);
        dstp+=1; srcp_m2+=1;srcp_m1+=1; srcp+=1; srcp_p1+=1;
    }          
    for(i=iterline ; --i ; ){ // last line
        *dstp = coeff[0]*(*srcp_m2) + coeff[1]*(*srcp_m1) + (coeff[2]+coeff[3]+coeff[4])*(*srcp);
        dstp+=1; srcp_m2+=1;srcp_m1+=1; srcp+=1; 
    }  
}

static void convolve_horiz_fast_3(image_t *dst, const image_t *src, const convolution_t *conv){
    const int stride_minus_1 = src->stride-1;
    const int iterline = (src->stride>>2);
    const float *coeff = conv->coeffs;
    v4sf *srcp = (v4sf*) src->c1, *dstp = (v4sf*) dst->c1;
    // create shifted version of src
    float *src_p1 = (float*) malloc(sizeof(float)*src->stride),
        *src_m1 = (float*) malloc(sizeof(float)*src->stride);
    int j;
    for(j=0;j<src->height;j++){
        int i;
        float *srcptr = (float*) srcp;
        const float right_coef = srcptr[src->width-1];
        for(i=src->width;i<src->stride;i++)
            srcptr[i] = right_coef;
        src_m1[0] = srcptr[0];
        memcpy(src_m1+1, srcptr , sizeof(float)*stride_minus_1);
        src_p1[stride_minus_1] = right_coef;
        memcpy(src_p1, srcptr+1, sizeof(float)*stride_minus_1);
        v4sf *srcp_p1 = (v4sf*) src_p1, *srcp_m1 = (v4sf*) src_m1;
        
        for(i=0;i<iterline;i++){
            *dstp = coeff[0]*(*srcp_m1) + coeff[1]*(*srcp) + coeff[2]*(*srcp_p1);
            dstp+=1; srcp_m1+=1; srcp+=1; srcp_p1+=1;
        }
    }
    free(src_p1);
    free(src_m1);
}

static void convolve_horiz_fast_5(image_t *dst, const image_t *src, const convolution_t *conv){
    const int stride_minus_1 = src->stride-1;
    const int stride_minus_2 = src->stride-2;
    const int iterline = (src->stride>>2);
    const float *coeff = conv->coeffs;
    v4sf *srcp = (v4sf*) src->c1, *dstp = (v4sf*) dst->c1;
    float *src_p1 = (float*) malloc(sizeof(float)*src->stride*4);
    float *src_p2 = src_p1+src->stride;
    float *src_m1 = src_p2+src->stride;
    float *src_m2 = src_m1+src->stride;
    int j;
    for(j=0;j<src->height;j++){
        int i;
        float *srcptr = (float*) srcp;
        const float right_coef = srcptr[src->width-1];
        for(i=src->width;i<src->stride;i++)
            srcptr[i] = right_coef;
        src_m1[0] = srcptr[0];
        memcpy(src_m1+1, srcptr , sizeof(float)*stride_minus_1);
        src_m2[0] = srcptr[0];
        src_m2[1] = srcptr[0];
        memcpy(src_m2+2, srcptr , sizeof(float)*stride_minus_2);
        src_p1[stride_minus_1] = right_coef;
        memcpy(src_p1, srcptr+1, sizeof(float)*stride_minus_1);
        src_p2[stride_minus_1] = right_coef;
        src_p2[stride_minus_2] = right_coef;
        memcpy(src_p2, srcptr+2, sizeof(float)*stride_minus_2);
                
        v4sf *srcp_p1 = (v4sf*) src_p1, *srcp_p2 = (v4sf*) src_p2, *srcp_m1 = (v4sf*) src_m1, *srcp_m2 = (v4sf*) src_m2;
        
        for(i=0;i<iterline;i++){
            *dstp = coeff[0]*(*srcp_m2) + coeff[1]*(*srcp_m1) + coeff[2]*(*srcp) + coeff[3]*(*srcp_p1) + coeff[4]*(*srcp_p2);
            dstp+=1; srcp_m2 +=1; srcp_m1+=1; srcp+=1; srcp_p1+=1; srcp_p2+=1;
        }
    }
    free(src_p1);
}

/* perform an horizontal convolution of an image */
void convolve_horiz(image_t *dest, const image_t *src, const convolution_t *conv){
    if(conv->order==1){
        convolve_horiz_fast_3(dest,src,conv);
        return;
    }else if(conv->order==2){
        convolve_horiz_fast_5(dest,src,conv);
        return;    
    }
    float *in = src->c1;
    float * out = dest->c1;
    int i, j, ii;
    float *o = out;
    int i0 = -conv->order;
    int i1 = +conv->order;
    float *coeff = conv->coeffs + conv->order;
    float *coeff_accu = conv->coeffs_accu + conv->order;
    for(j = 0; j < src->height; j++){
        const float *al = in + j * src->stride;
        const float *f0 = coeff + i0;
        float sum;
        for(i = 0; i < -i0; i++){
	        sum=coeff_accu[-i - 1] * al[0];
	        for(ii = i1 + i; ii >= 0; ii--){
	            sum += coeff[ii - i] * al[ii];
            }
	        *o++ = sum;
        }
        for(; i < src->width - i1; i++){
	        sum = 0;
	        for(ii = i1 - i0; ii >= 0; ii--){
	            sum += f0[ii] * al[ii];
            }
	        al++;
	        *o++ = sum;
        }
        for(; i < src->width; i++){
	        sum = coeff_accu[src->width - i] * al[src->width - i0 - 1 - i];
	        for(ii = src->width - i0 - 1 - i; ii >= 0; ii--){
	            sum += f0[ii] * al[ii];
            }
	        al++;
	        *o++ = sum;
        }
        for(i = 0; i < src->stride - src->width; i++){
	        o++;
        }
    }
}

/* perform a vertical convolution of an image */
void convolve_vert(image_t *dest, const image_t *src, const convolution_t *conv){
    if(conv->order==1){
        convolve_vert_fast_3(dest,src,conv);
        return;
    }else if(conv->order==2){
        convolve_vert_fast_5(dest,src,conv);
        return;    
    }
    float *in = src->c1;
    float *out = dest->c1;
    int i0 = -conv->order;
    int i1 = +conv->order;
    float *coeff = conv->coeffs + conv->order;
    float *coeff_accu = conv->coeffs_accu + conv->order;
    int i, j, ii;
    float *o = out;
    const float *alast = in + src->stride * (src->height - 1);
    const float *f0 = coeff + i0;
    for(i = 0; i < -i0; i++){
        float fa = coeff_accu[-i - 1];
        const float *al = in + i * src->stride;
        for(j = 0; j < src->width; j++){
	        float sum = fa * in[j];
	        for(ii = -i; ii <= i1; ii++){
	            sum += coeff[ii] * al[j + ii * src->stride];
            }
	        *o++ = sum;
        }
        for(j = 0; j < src->stride - src->width; j++) 
	    {
	        o++;
        }
    }
    for(; i < src->height - i1; i++){
        const float *al = in + (i + i0) * src->stride;
        for(j = 0; j < src->width; j++){
	        float sum = 0;
	        const float *al2 = al;
	        for(ii = 0; ii <= i1 - i0; ii++){
	            sum += f0[ii] * al2[0];
	            al2 += src->stride;
            }
	        *o++ = sum;
	        al++;
        }
        for(j = 0; j < src->stride - src->width; j++){
	        o++;
        }
    }
    for(;i < src->height; i++){
        float fa = coeff_accu[src->height - i];
        const float *al = in + i * src->stride;
        for(j = 0; j < src->width; j++){
	        float sum = fa * alast[j];
	        for(ii = i0; ii <= src->height - 1 - i; ii++){
	            sum += coeff[ii] * al[j + ii * src->stride];
            }
	        *o++ = sum;
        }
        for(j = 0; j < src->stride - src->width; j++){
	        o++;
        }
    }
}

/* free memory of a convolution structure */
void convolution_delete(convolution_t *conv){
    if(conv)
    {
        free(conv->coeffs);
        free(conv->coeffs_accu);
        free(conv);
    }
}

/* perform horizontal and/or vertical convolution to a color image */
void color_image_convolve_hv(color_image_t *dst, const color_image_t *src, const convolution_t *horiz_conv, const convolution_t *vert_conv){
    const int width = src->width, height = src->height, stride = src->stride;
    // separate channels of images
    image_t src_red = {width,height,stride,src->c1}, src_green = {width,height,stride,src->c2}, src_blue = {width,height,stride,src->c3}, 
            dst_red = {width,height,stride,dst->c1}, dst_green = {width,height,stride,dst->c2}, dst_blue = {width,height,stride,dst->c3};
    // horizontal and vertical
    if(horiz_conv != NULL && vert_conv != NULL){
        float *tmp_data = malloc(sizeof(float)*stride*height);
        if(tmp_data == NULL){
	        fprintf(stderr,"error color_image_convolve_hv(): not enough memory\n");
	        exit(1);
        }  
        image_t tmp = {width,height,stride,tmp_data};   
        // perform convolution for each channel
        convolve_horiz(&tmp,&src_red,horiz_conv); 
        convolve_vert(&dst_red,&tmp,vert_conv); 
        convolve_horiz(&tmp,&src_green,horiz_conv);
        convolve_vert(&dst_green,&tmp,vert_conv); 
        convolve_horiz(&tmp,&src_blue,horiz_conv); 
        convolve_vert(&dst_blue,&tmp,vert_conv);
        free(tmp_data);
    }else if(horiz_conv != NULL && vert_conv == NULL){ // only horizontal
        convolve_horiz(&dst_red,&src_red,horiz_conv);
        convolve_horiz(&dst_green,&src_green,horiz_conv);
        convolve_horiz(&dst_blue,&src_blue,horiz_conv);
    }else if(vert_conv != NULL && horiz_conv == NULL){ // only vertical
        convolve_vert(&dst_red,&src_red,vert_conv);
        convolve_vert(&dst_green,&src_green,vert_conv);
        convolve_vert(&dst_blue,&src_blue,vert_conv);
    }
}

/* perform horizontal and/or vertical convolution to a single band image*/
void image_convolve_hv(image_t *dst, const image_t *src, const convolution_t *horiz_conv, const convolution_t *vert_conv)
{
    const int width = src->width, height = src->height, stride = src->stride;
    // separate channels of images
    image_t src_red = {width,height,stride,src->c1}, 
            dst_red = {width,height,stride,dst->c1};
    // horizontal and vertical
    if(horiz_conv != NULL && vert_conv != NULL){
        float *tmp_data = malloc(sizeof(float)*stride*height);
        if(tmp_data == NULL){
          fprintf(stderr,"error image_convolve_hv(): not enough memory\n");
          exit(1);
        }  
        image_t tmp = {width,height,stride,tmp_data};   
        // perform convolution for each channel
        convolve_horiz(&tmp,&src_red,horiz_conv); 
        convolve_vert(&dst_red,&tmp,vert_conv); 
        free(tmp_data);
    }else if(horiz_conv != NULL && vert_conv == NULL){ // only horizontal
        convolve_horiz(&dst_red,&src_red,horiz_conv);
    }else if(vert_conv != NULL && horiz_conv == NULL){ // only vertical
        convolve_vert(&dst_red,&src_red,vert_conv);
    }
}

/************ Pyramid **********/

/* create new color image pyramid structures */
static color_image_pyramid_t* color_image_pyramid_new(){
    color_image_pyramid_t* pyr = (color_image_pyramid_t*) malloc(sizeof(color_image_pyramid_t));
    if(pyr == NULL){
        fprintf(stderr,"Error in color_image_pyramid_new(): not enough memory\n");
        exit(1);
    }
    pyr->min_size = -1;
    pyr->scale_factor = -1.0f;
    pyr->size = -1;
    pyr->images = NULL;
    return pyr;
}

/* set the size of the color image pyramid structures (reallocate the array of pointers to images) */
static void color_image_pyramid_set_size(color_image_pyramid_t* pyr, const int size){
    if(size<0){
        fprintf(stderr,"Error in color_image_pyramid_set_size(): size is negative\n");
        exit(1);
    }
    if(pyr->images == NULL){
        pyr->images = (color_image_t**) malloc(sizeof(color_image_t*)*size);
    }else{
        pyr->images = (color_image_t**) realloc(pyr->images,sizeof(color_image_t*)*size);
    }
    if(pyr->images == NULL){
        fprintf(stderr,"Error in color_image_pyramid_set_size(): not enough memory\n");
        exit(1);      
    }
    pyr->size = size;
}

/* create a pyramid of color images using a given scale factor, stopping when one dimension reach min_size and with applying a gaussian smoothing of standard deviation spyr (no smoothing if 0) */
color_image_pyramid_t *color_image_pyramid_create(const color_image_t *src, const float scale_factor, const int min_size, const float spyr){
    const int nb_max_scale = 1000;
    // allocate structure
    color_image_pyramid_t *pyramid = color_image_pyramid_new();
    pyramid->min_size = min_size;
    pyramid->scale_factor = scale_factor;
    convolution_t *conv = NULL;
    if(spyr>0.0f){
        int fsize;
        float *filter_coef = gaussian_filter(spyr, &fsize);
        conv = convolution_new(fsize, filter_coef, 1);
        free(filter_coef);
    }
    color_image_pyramid_set_size(pyramid, nb_max_scale);
    pyramid->images[0] = color_image_cpy(src);
    int i;
    for( i=1 ; i<nb_max_scale ; i++){
        const int oldwidth = pyramid->images[i-1]->width, oldheight = pyramid->images[i-1]->height;
        const int newwidth = (int) (1.5f + (oldwidth-1) / scale_factor);
        const int newheight = (int) (1.5f + (oldheight-1) / scale_factor);
        if( newwidth <= min_size || newheight <= min_size){
            color_image_pyramid_set_size(pyramid, i);
	        break;
	    }
        if(spyr>0.0f){
            color_image_t* tmp = color_image_new(oldwidth, oldheight);
	        color_image_convolve_hv(tmp,pyramid->images[i-1], conv, conv);
	        pyramid->images[i]= color_image_resize_bilinear(tmp, scale_factor);
	        color_image_delete(tmp);
	    }else{
	        pyramid->images[i] = color_image_resize_bilinear(pyramid->images[i-1], scale_factor);
	    }
    }
    if(spyr>0.0f){
        convolution_delete(conv);
    }
    return pyramid;
}

/* delete the structure of a pyramid of color images and all the color images in it*/
void color_image_pyramid_delete(color_image_pyramid_t *pyr){
    if(pyr==NULL){
        return;
    }
    int i;
    for(i=0 ; i<pyr->size ; i++){
        color_image_delete(pyr->images[i]);
    }
    free(pyr->images);
    free(pyr);
}
