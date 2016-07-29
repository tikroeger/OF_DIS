#ifndef VARREF_HEADER
#define VARREF_HEADER

#include "FDF1.0.1/image.h"
#include "FDF1.0.1/opticalflow_aux.h"
#include "FDF1.0.1/solver.h"

#include "oflow.h"

namespace OFC
{

typedef __v4sf v4sf;

typedef struct
{
  float alpha;             // smoothness weight
  float beta;              // matching weight
  float gamma;             // gradient constancy assumption weight
  float delta;             // color constancy assumption weight
  int n_inner_iteration;   // number of inner fixed point iterations
  int n_solver_iteration;  // number of solver iterations 
  float sor_omega;         // omega parameter of sor method
  
  float tmp_quarter_alpha;
  float tmp_half_gamma_over3;
  float tmp_half_delta_over3;
  float tmp_half_beta;
  
} TVparams;  
        
  
class VarRefClass
{
  
public:
  VarRefClass(const float * im_ao_in, const float * im_ao_dx_in, const float * im_ao_dy_in, // expects #sc_f_in pointers to float arrays for images and gradients. 
              const float * im_bo_in, const float * im_bo_dx_in, const float * im_bo_dy_in,
              const camparam* cpt_in, const camparam* cpo_in,const optparam* op_in, float *flowout);
  ~VarRefClass();  

private:

  convolution_t *deriv, *deriv_flow;
  

  #if (SELECTCHANNEL==1 | SELECTCHANNEL==2)    // Intensity image, or gradient image
  void copyimage(const float* img, image_t * img_t);
  void RefLevelOF(image_t *wx, image_t *wy, const image_t *im1, const image_t *im2);
  void RefLevelDE(image_t *wx, const image_t *im1, const image_t *im2);
  #else // 3-Color RGB image
  void copyimage(const float* img, color_image_t * img_t);
  void RefLevelOF(image_t *wx, image_t *wy, const color_image_t *im1, const color_image_t *im2);
  void RefLevelDE(image_t *wx, const color_image_t *im1, const color_image_t *im2);    
  #endif
  
  TVparams tvparams;

  const camparam* cpt;
  const camparam* cpo;
  const optparam* op;    
    
};

}

#endif /* VARREF_HEADER */


