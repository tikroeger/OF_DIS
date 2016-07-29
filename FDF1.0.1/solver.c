#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <malloc.h>

#include <omp.h>

#include "image.h"
#include "solver.h"

#include <xmmintrin.h>
typedef __v4sf v4sf;

//THIS IS A SLOW VERSION BUT READABLE
//Perform n iterations of the sor_coupled algorithm
//du and dv are used as initial guesses
//The system form is the same as in opticalflow.c
void sor_coupled_slow_but_readable(image_t *du, image_t *dv, image_t *a11, image_t *a12, image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, const int iterations, const float omega)
{  
    int i,j,iter;
    for(iter = 0 ; iter<iterations ; iter++)
    {
    #pragma omp parallel for
    for(j=0 ; j<du->height ; j++)
    {
      float sigma_u,sigma_v,sum_dpsis,A11,A22,A12,B1,B2;//,det;
      for(i=0 ; i<du->width ; i++)
      {
          sigma_u = 0.0f;
          sigma_v = 0.0f;
          sum_dpsis = 0.0f;
          if(j>0)
          {
            sigma_u -= dpsis_vert->c1[(j-1)*du->stride+i]*du->c1[(j-1)*du->stride+i];
            sigma_v -= dpsis_vert->c1[(j-1)*du->stride+i]*dv->c1[(j-1)*du->stride+i];
            sum_dpsis += dpsis_vert->c1[(j-1)*du->stride+i];
          }
          if(i>0)
          {
            sigma_u -= dpsis_horiz->c1[j*du->stride+i-1]*du->c1[j*du->stride+i-1];
            sigma_v -= dpsis_horiz->c1[j*du->stride+i-1]*dv->c1[j*du->stride+i-1];
            sum_dpsis += dpsis_horiz->c1[j*du->stride+i-1];
          }
          if(j<du->height-1)
          {
            sigma_u -= dpsis_vert->c1[j*du->stride+i]*du->c1[(j+1)*du->stride+i];
            sigma_v -= dpsis_vert->c1[j*du->stride+i]*dv->c1[(j+1)*du->stride+i];
            sum_dpsis += dpsis_vert->c1[j*du->stride+i];
          }
          if(i<du->width-1)
          {
            sigma_u -= dpsis_horiz->c1[j*du->stride+i]*du->c1[j*du->stride+i+1];
            sigma_v -= dpsis_horiz->c1[j*du->stride+i]*dv->c1[j*du->stride+i+1];
            sum_dpsis += dpsis_horiz->c1[j*du->stride+i];
          }
          A11 = a11->c1[j*du->stride+i]+sum_dpsis;
          A12 = a12->c1[j*du->stride+i];
          A22 = a22->c1[j*du->stride+i]+sum_dpsis;
          //det = A11*A22-A12*A12;
          B1 = b1->c1[j*du->stride+i]-sigma_u;
          B2 = b2->c1[j*du->stride+i]-sigma_v;
//           du->c1[j*du->stride+i] = (1.0f-omega)*du->c1[j*du->stride+i] +omega*( A22*B1-A12*B2)/det;
//           dv->c1[j*du->stride+i] = (1.0f-omega)*dv->c1[j*du->stride+i] +omega*(-A12*B1+A11*B2)/det;
          du->c1[j*du->stride+i] = (1.0f-omega)*du->c1[j*du->stride+i] + omega/A11 *(B1 - A12* dv->c1[j*du->stride+i] );
          dv->c1[j*du->stride+i] = (1.0f-omega)*dv->c1[j*du->stride+i] + omega/A22 *(B2 - A12* du->c1[j*du->stride+i] );
          
          
      }
    }
  }  
}

 // THIS IS A FASTER VERSION BUT UNREADABLE, ONLY OPTICAL FLOW WITHOUT OPENMP PARALLELIZATION
 // the first iteration is separated from the other to compute the inverse of the 2x2 block diagonal
 // each iteration is split in two first line / middle lines / last line, and the left block is computed separately on each line
void sor_coupled(image_t *du, image_t *dv, image_t *a11, image_t *a12, image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, const int iterations, const float omega){
    //sor_coupled_slow(du,dv,a11,a12,a22,b1,b2,dpsis_horiz,dpsis_vert,iterations,omega); return; printf("test\n");
  
    if(du->width<2 || du->height<2 || iterations < 1){
        sor_coupled_slow_but_readable(du,dv,a11,a12,a22,b1,b2,dpsis_horiz,dpsis_vert,iterations,omega);
        return;
    }
    
    const int stride = du->stride, width = du->width;
    const int iterheight = du->height-1, iterline = (stride)/4, width_minus_1_sizeoffloat = sizeof(float)*(width-1);
    int j,iter,i,k;
    float *floatarray = (float*) memalign(16, stride*sizeof(float)*3); 
    if(floatarray==NULL){
        fprintf(stderr, "error in sor_coupled(): not enough memory\n");
        exit(1);
    }   
    float *f1 = floatarray;
    float *f2 = f1+stride;
    float *f3 = f2+stride;
    f1[0] = 0.0f;
    memset(&f1[width], 0, sizeof(float)*(stride-width));
    memset(&f2[width-1], 0, sizeof(float)*(stride-width+1));
    memset(&f3[width-1], 0, sizeof(float)*(stride-width+1));   	  
    	  
    { // first iteration
        v4sf *a11p = (v4sf*) a11->c1, *a12p = (v4sf*) a12->c1, *a22p = (v4sf*) a22->c1, *b1p = (v4sf*) b1->c1, *b2p = (v4sf*) b2->c1, *hp = (v4sf*) dpsis_horiz->c1, *vp = (v4sf*) dpsis_vert->c1;
        float *du_ptr = du->c1, *dv_ptr = dv->c1;
        v4sf *dub = (v4sf*) (du_ptr+stride), *dvb = (v4sf*) (dv_ptr+stride);
        
        { // first iteration - first line
        
            memcpy(f1+1, ((float*) hp), width_minus_1_sizeoffloat);   
            memcpy(f2, du_ptr+1, width_minus_1_sizeoffloat);
            memcpy(f3, dv_ptr+1, width_minus_1_sizeoffloat);
            v4sf* hpl = (v4sf*) f1, *dur = (v4sf*) f2, *dvr = (v4sf*) f3;
            
            { // left block
                // reverse 2x2 diagonal block
                const v4sf dpsis = (*hpl) + (*hp) + (*vp);
                const v4sf A11 = (*a22p)+dpsis, A22 = (*a11p)+dpsis;
                const v4sf det = A11*A22 - (*a12p)*(*a12p);
                *a11p = A11/det;
                *a22p = A22/det;
                *a12p /= -det;
                // do one iteration
                const v4sf s1 = (*hp)*(*dur) + (*vp)*(*dub) + (*b1p);
                const v4sf s2 = (*hp)*(*dvr) + (*vp)*(*dvb) + (*b2p);
                du_ptr[0] += omega*( a11p[0][0]*s1[0] + a12p[0][0]*s2[0] - du_ptr[0] );
	            dv_ptr[0] += omega*( a12p[0][0]*s1[0] + a22p[0][0]*s2[0] - dv_ptr[0] );             
                for(k=1;k<4;k++){
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vp+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dub+=1; dvb +=1; b1p+=1; b2p+=1;
                du_ptr += 4; dv_ptr += 4;        
            }
            for(i=iterline;--i;){
                // reverse 2x2 diagonal block
                const v4sf dpsis = (*hpl) + (*hp) + (*vp);
                const v4sf A11 = (*a22p)+dpsis, A22 = (*a11p)+dpsis;
                const v4sf det = A11*A22 - (*a12p)*(*a12p);
                *a11p = A11/det;
                *a22p = A22/det;
                *a12p /= -det;
                // do one iteration
                const v4sf s1 = (*hp)*(*dur) + (*vp)*(*dub) + (*b1p);
                const v4sf s2 = (*hp)*(*dvr) + (*vp)*(*dvb) + (*b2p);
                for(k=0;k<4;k++){
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vp+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dub+=1; dvb +=1; b1p+=1; b2p+=1;
                du_ptr += 4; dv_ptr += 4;
            }
          
        }
        
        v4sf *vpt = (v4sf*) dpsis_vert->c1;
        v4sf *dut = (v4sf*) du->c1, *dvt = (v4sf*) dv->c1;
        
        for(j=iterheight;--j;){ // first iteration - middle lines
            memcpy(f1+1, ((float*) hp), width_minus_1_sizeoffloat);   
            memcpy(f2, du_ptr+1, width_minus_1_sizeoffloat);
            memcpy(f3, dv_ptr+1, width_minus_1_sizeoffloat);
            v4sf* hpl = (v4sf*) f1, *dur = (v4sf*) f2, *dvr = (v4sf*) f3;
                 
            { // left block
                // reverse 2x2 diagonal block
                const v4sf dpsis = (*hpl) + (*hp) + (*vpt) + (*vp);
                const v4sf A11 = (*a22p)+dpsis, A22 = (*a11p)+dpsis;
                const v4sf det = A11*A22 - (*a12p)*(*a12p);
                *a11p = A11/det;
                *a22p = A22/det;
                *a12p /= -det;
                // do one iteration
                const v4sf s1 = (*hp)*(*dur) + (*vpt)*(*dut) + (*vp)*(*dub) + (*b1p);
                const v4sf s2 = (*hp)*(*dvr) + (*vpt)*(*dvt) + (*vp)*(*dvb) + (*b2p);
                du_ptr[0] += omega*( a11p[0][0]*s1[0] + a12p[0][0]*s2[0] - du_ptr[0] );
	            dv_ptr[0] += omega*( a12p[0][0]*s1[0] + a22p[0][0]*s2[0] - dv_ptr[0] );             
                for(k=1;k<4;k++){
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vpt+=1; vp+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dut+=1; dvt+=1; dub+=1; dvb +=1; b1p+=1; b2p+=1;
                du_ptr += 4; dv_ptr += 4;           
            }
            for(i=iterline;--i;){
                // reverse 2x2 diagonal block
                const v4sf dpsis = (*hpl) + (*hp) + (*vpt) + (*vp);
                const v4sf A11 = (*a22p)+dpsis, A22 = (*a11p)+dpsis;
                const v4sf det = A11*A22 - (*a12p)*(*a12p);
                *a11p = A11/det;
                *a22p = A22/det;
                *a12p /= -det;
                // do one iteration
                const v4sf s1 = (*hp)*(*dur) + (*vpt)*(*dut) + (*vp)*(*dub) + (*b1p);
                const v4sf s2 = (*hp)*(*dvr) + (*vpt)*(*dvt) + (*vp)*(*dvb) + (*b2p);
                for(k=0;k<4;k++){
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vpt+=1; vp+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dut+=1; dvt+=1; dub+=1; dvb +=1; b1p+=1; b2p+=1;
                du_ptr += 4; dv_ptr += 4;
            }
                
        }
        
        { // first iteration - last line
            memcpy(f1+1, ((float*) hp), width_minus_1_sizeoffloat);   
            memcpy(f2, du_ptr+1, width_minus_1_sizeoffloat);
            memcpy(f3, dv_ptr+1, width_minus_1_sizeoffloat);
            v4sf* hpl = (v4sf*) f1, *dur = (v4sf*) f2, *dvr = (v4sf*) f3;

            { // left block
                // reverse 2x2 diagonal block
                const v4sf dpsis = (*hpl) + (*hp) + (*vpt);
                const v4sf A11 = (*a22p)+dpsis, A22 = (*a11p)+dpsis;
                const v4sf det = A11*A22 - (*a12p)*(*a12p);
                *a11p = A11/det;
                *a22p = A22/det;
                *a12p /= -det;
                // do one iteration
                const v4sf s1 = (*hp)*(*dur) + (*vpt)*(*dut) + (*b1p);
                const v4sf s2 = (*hp)*(*dvr) + (*vpt)*(*dvt) + (*b2p);
                du_ptr[0] += omega*( a11p[0][0]*s1[0] + a12p[0][0]*s2[0] - du_ptr[0] );
	            dv_ptr[0] += omega*( a12p[0][0]*s1[0] + a22p[0][0]*s2[0] - dv_ptr[0] );             
                for(k=1;k<4;k++){
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vpt+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dut+=1; dvt+=1; b1p+=1; b2p+=1;
                du_ptr += 4; dv_ptr += 4;           
            }
            for(i=iterline;--i;){
                // reverse 2x2 diagonal block
                const v4sf dpsis = (*hpl) + (*hp) + (*vpt);
                const v4sf A11 = (*a22p)+dpsis, A22 = (*a11p)+dpsis;
                const v4sf det = A11*A22 - (*a12p)*(*a12p);
                *a11p = A11/det;
                *a22p = A22/det;
                *a12p /= -det;
                // do one iteration
                const v4sf s1 = (*hp)*(*dur) + (*vpt)*(*dut) + (*b1p);
                const v4sf s2 = (*hp)*(*dvr) + (*vpt)*(*dvt) + (*b2p);
                for(k=0;k<4;k++){
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vpt+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dut+=1; dvt+=1; b1p+=1; b2p+=1;
                du_ptr += 4; dv_ptr += 4;
            }

        }
    }

    

   for(iter=iterations;--iter;)   // other iterations
   {
        v4sf *a11p = (v4sf*) a11->c1, *a12p = (v4sf*) a12->c1, *a22p = (v4sf*) a22->c1, *b1p = (v4sf*) b1->c1, *b2p = (v4sf*) b2->c1, *hp = (v4sf*) dpsis_horiz->c1, *vp = (v4sf*) dpsis_vert->c1;
        float *du_ptr = du->c1, *dv_ptr = dv->c1;
        v4sf *dub = (v4sf*) (du_ptr+stride), *dvb = (v4sf*) (dv_ptr+stride);
        
        { // other iteration - first line
        
            memcpy(f1+1, ((float*) hp), width_minus_1_sizeoffloat);   
            memcpy(f2, du_ptr+1, width_minus_1_sizeoffloat);
            memcpy(f3, dv_ptr+1, width_minus_1_sizeoffloat);
            v4sf* hpl = (v4sf*) f1, *dur = (v4sf*) f2, *dvr = (v4sf*) f3;
            
            { // left block
                // do one iteration
                const v4sf s1 = (*hp)*(*dur) + (*vp)*(*dub) + (*b1p);
                const v4sf s2 = (*hp)*(*dvr) + (*vp)*(*dvb) + (*b2p);
                du_ptr[0] += omega*( a11p[0][0]*s1[0] + a12p[0][0]*s2[0] - du_ptr[0] );
	            dv_ptr[0] += omega*( a12p[0][0]*s1[0] + a22p[0][0]*s2[0] - dv_ptr[0] );             
                for(k=1;k<4;k++){
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vp+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dub+=1; dvb +=1; b1p+=1; b2p+=1;
                du_ptr += 4; dv_ptr += 4;        
            }
            for(i=iterline;--i;){
                // do one iteration
                const v4sf s1 = (*hp)*(*dur) + (*vp)*(*dub) + (*b1p);
                const v4sf s2 = (*hp)*(*dvr) + (*vp)*(*dvb) + (*b2p);
                for(k=0;k<4;k++){
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vp+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dub+=1; dvb +=1; b1p+=1; b2p+=1;
                du_ptr += 4; dv_ptr += 4;
            }
          
        }
        
        v4sf *vpt = (v4sf*) dpsis_vert->c1;
        v4sf *dut = (v4sf*) du->c1, *dvt = (v4sf*) dv->c1;

        for(j=iterheight;--j;)  // other iteration - middle lines
	{ 
            memcpy(f1+1, ((float*) hp), width_minus_1_sizeoffloat);   
            memcpy(f2, du_ptr+1, width_minus_1_sizeoffloat);
            memcpy(f3, dv_ptr+1, width_minus_1_sizeoffloat);
            v4sf* hpl = (v4sf*) f1, *dur = (v4sf*) f2, *dvr = (v4sf*) f3;
                 
            { // left block
                // do one iteration
                const v4sf s1 = (*hp)*(*dur) + (*vpt)*(*dut) + (*vp)*(*dub) + (*b1p);
                const v4sf s2 = (*hp)*(*dvr) + (*vpt)*(*dvt) + (*vp)*(*dvb) + (*b2p);
                du_ptr[0] += omega*( a11p[0][0]*s1[0] + a12p[0][0]*s2[0] - du_ptr[0] );
		dv_ptr[0] += omega*( a12p[0][0]*s1[0] + a22p[0][0]*s2[0] - dv_ptr[0] );             
                for(k=1;k<4;k++)
		{
		  const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
		  const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
		  du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
		  dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vpt+=1; vp+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dut+=1; dvt+=1; dub+=1; dvb +=1; b1p+=1; b2p+=1;
                du_ptr += 4; dv_ptr += 4;           
            }
            
            for(i=iterline; --i;)
	    {
	      // do one iteration
	      const v4sf s1 = (*hp)*(*dur) + (*vpt)*(*dut) + (*vp)*(*dub) + (*b1p);
	      const v4sf s2 = (*hp)*(*dvr) + (*vpt)*(*dvt) + (*vp)*(*dvb) + (*b2p);
	      for(k=0;k<4;k++)
	      {
		  const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
		  const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
		  du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
		      dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
	      }
	      // increment pointer
	      hpl+=1; hp+=1; vpt+=1; vp+=1; a11p+=1; a12p+=1; a22p+=1;
	      dur+=1; dvr+=1; dut+=1; dvt+=1; dub+=1; dvb +=1; b1p+=1; b2p+=1;
	      du_ptr += 4; dv_ptr += 4;
            }
                
        }
        
        { // other iteration - last line
            memcpy(f1+1, ((float*) hp), width_minus_1_sizeoffloat);   
            memcpy(f2, du_ptr+1, width_minus_1_sizeoffloat);
            memcpy(f3, dv_ptr+1, width_minus_1_sizeoffloat);
            v4sf* hpl = (v4sf*) f1, *dur = (v4sf*) f2, *dvr = (v4sf*) f3;

            { // left block
                // do one iteration
                const v4sf s1 = (*hp)*(*dur) + (*vpt)*(*dut) + (*b1p);
                const v4sf s2 = (*hp)*(*dvr) + (*vpt)*(*dvt) + (*b2p);
                du_ptr[0] += omega*( a11p[0][0]*s1[0] + a12p[0][0]*s2[0] - du_ptr[0] );
	            dv_ptr[0] += omega*( a12p[0][0]*s1[0] + a22p[0][0]*s2[0] - dv_ptr[0] );             
                for(k=1;k<4;k++){
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vpt+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dut+=1; dvt+=1; b1p+=1; b2p+=1;
                du_ptr += 4; dv_ptr += 4;           
            }
            for(i=iterline;--i;){
                // do one iteration
                const v4sf s1 = (*hp)*(*dur) + (*vpt)*(*dut) + (*b1p);
                const v4sf s2 = (*hp)*(*dvr) + (*vpt)*(*dvt) + (*b2p);
                for(k=0;k<4;k++){
                    const float B1 = hpl[0][k]*du_ptr[k-1] + s1[k];
                    const float B2 = hpl[0][k]*dv_ptr[k-1] + s2[k];
                    du_ptr[k] += omega*( a11p[0][k]*B1 + a12p[0][k]*B2 - du_ptr[k] );
	                dv_ptr[k] += omega*( a12p[0][k]*B1 + a22p[0][k]*B2 - dv_ptr[k] );
                }
                // increment pointer
                hpl+=1; hp+=1; vpt+=1; a11p+=1; a12p+=1; a22p+=1;
                dur+=1; dvr+=1; dut+=1; dvt+=1; b1p+=1; b2p+=1;
                du_ptr += 4; dv_ptr += 4;
            }

        }
    }



    free(floatarray);

}


//THIS IS A SLOW VERSION BUT READABLE
//Perform n iterations of the sor_coupled algorithm
//du is used as initial guesses
//The system form is the same as in opticalflow.c
void sor_coupled_slow_but_readable_DE(image_t *du, const image_t *a11, const image_t *b1, const image_t *dpsis_horiz, const image_t *dpsis_vert, const int iterations, const float omega)
{
    int i,j,iter;
    for(iter = 0 ; iter<iterations ; iter++)
    {
	#pragma omp parallel for
        for(j=0 ; j<du->height ; j++)
	{
	  float sigma_u,sum_dpsis,A11,B1;
	        for(i=0 ; i<du->width ; i++){
	            sigma_u = 0.0f;
	            sum_dpsis = 0.0f;
	            if(j>0)
		    {
		      sigma_u -= dpsis_vert->c1[(j-1)*du->stride+i]*du->c1[(j-1)*du->stride+i];
		      sum_dpsis += dpsis_vert->c1[(j-1)*du->stride+i];
		    }
	            if(i>0)
		    {
		      sigma_u -= dpsis_horiz->c1[j*du->stride+i-1]*du->c1[j*du->stride+i-1];
		      sum_dpsis += dpsis_horiz->c1[j*du->stride+i-1];
		    }
	            if(j<du->height-1)
		    {
		      sigma_u -= dpsis_vert->c1[j*du->stride+i]*du->c1[(j+1)*du->stride+i];
		      sum_dpsis += dpsis_vert->c1[j*du->stride+i];
		    }
	            if(i<du->width-1)
		    {
		      sigma_u -= dpsis_horiz->c1[j*du->stride+i]*du->c1[j*du->stride+i+1];
		      sum_dpsis += dpsis_horiz->c1[j*du->stride+i];
		    }
                A11 = a11->c1[j*du->stride+i]+sum_dpsis;
                B1 = b1->c1[j*du->stride+i]-sigma_u;
                du->c1[j*du->stride+i] = (1.0f-omega)*du->c1[j*du->stride+i] +omega*( B1/A11 );
	        }
	    }
    }
}



