
// #include <opencv2/core/core.hpp> // needed for verbosity >= 3, DISVISUAL
// #include <opencv2/highgui/highgui.hpp> // needed for verbosity >= 3, DISVISUAL
// #include <opencv2/imgproc/imgproc.hpp> // needed for verbosity >= 3, DISVISUAL

#include <iostream>
#include <string>
#include <vector>
#include <valarray>

#include <thread>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <stdio.h>  

#include "patch.h"
#include "patchgrid.h"


using std::cout;
using std::endl;
using std::vector;


namespace OFC
{
    
  PatGridClass::PatGridClass(
    const camparam* cpt_in,
    const camparam* cpo_in,
    const optparam* op_in)
  : 
    cpt(cpt_in),
    cpo(cpo_in),
    op(op_in)
  {

  // Generate grid on current scale
  steps = op->steps;
  nopw = ceil( (float)cpt->width /  (float)steps );
  noph = ceil( (float)cpt->height / (float)steps );
  const int offsetw = floor((cpt->width - (nopw-1)*steps)/2);
  const int offseth = floor((cpt->height - (noph-1)*steps)/2);

  nopatches = nopw*noph;
  pt_ref.resize(nopatches);
  p_init.resize(nopatches);
  pat.reserve(nopatches);
  
  im_ao_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);
  im_ao_dx_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);
  im_ao_dy_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);

  im_bo_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);
  im_bo_dx_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);
  im_bo_dy_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);

  int patchid=0;
  for (int x = 0; x < nopw; ++x)
  {
    for (int y = 0; y < noph; ++y)
    {
      int i = x*noph + y;

      pt_ref[i][0] = x * steps + offsetw;
      pt_ref[i][1] = y * steps + offseth;
      p_init[i].setZero();
      
      pat.push_back(new OFC::PatClass(cpt, cpo, op, patchid));    
      patchid++;
    }
  }
}

PatGridClass::~PatGridClass()
{
  delete im_ao_eg;
  delete im_ao_dx_eg;
  delete im_ao_dy_eg;

  delete im_bo_eg;
  delete im_bo_dx_eg;
  delete im_bo_dy_eg;

  for (int i=0; i< nopatches; ++i)
    delete pat[i];
}

void PatGridClass::SetComplGrid(PatGridClass *cg_in)
{
  cg = cg_in;
}


void PatGridClass::InitializeGrid(const float * im_ao_in, const float * im_ao_dx_in, const float * im_ao_dy_in)
{
  im_ao = im_ao_in;
  im_ao_dx = im_ao_dx_in;
  im_ao_dy = im_ao_dy_in;
  
  new (im_ao_eg) Eigen::Map<const Eigen::MatrixXf>(im_ao,cpt->height,cpt->width); // new placement operator
  new (im_ao_dx_eg) Eigen::Map<const Eigen::MatrixXf>(im_ao_dx,cpt->height,cpt->width);  
  new (im_ao_dy_eg) Eigen::Map<const Eigen::MatrixXf>(im_ao_dy,cpt->height,cpt->width);  
  
  
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nopatches; ++i)
  {
    pat[i]->InitializePatch(im_ao_eg, im_ao_dx_eg, im_ao_dy_eg, pt_ref[i]);
    p_init[i].setZero();    
  }

}

void PatGridClass::SetTargetImage(const float * im_bo_in, const float * im_bo_dx_in, const float * im_bo_dy_in)
{
  im_bo = im_bo_in;
  im_bo_dx = im_bo_dx_in;
  im_bo_dy = im_bo_dy_in;
  
  new (im_bo_eg) Eigen::Map<const Eigen::MatrixXf>(im_bo,cpt->height,cpt->width); // new placement operator
  new (im_bo_dx_eg) Eigen::Map<const Eigen::MatrixXf>(im_bo_dx,cpt->height,cpt->width); // new placement operator
  new (im_bo_dy_eg) Eigen::Map<const Eigen::MatrixXf>(im_bo_dy,cpt->height,cpt->width); // new placement operator
  
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nopatches; ++i)
    pat[i]->SetTargetImage(im_bo_eg, im_bo_dx_eg, im_bo_dy_eg);
  
}

void PatGridClass::Optimize()
{
    #pragma omp parallel for schedule(dynamic,10)
    for (int i = 0; i < nopatches; ++i)
    {
      pat[i]->OptimizeIter(p_init[i], true); // optimize until convergence  
    }
}  

// void PatGridClass::OptimizeAndVisualize(const float sc_fct_tmp) // needed for verbosity >= 3, DISVISUAL
// {
//   bool allconverged=0;
//   int cnt = 0;
//   while (!allconverged)
//   {
//     cnt++;
// 
//     allconverged=1;
// 
//     for (int i = 0; i < nopatches; ++i)
//     {
//       if (pat[i]->isConverged()==0)
//       {
//         pat[i]->OptimizeIter(p_init[i], false); // optimize, only one iterations
//         allconverged=0;
//       }
//     }
//     
// 
//     // Display original image
//     const cv::Mat src(cpt->height+2*cpt->imgpadding, cpt->width+2*cpt->imgpadding, CV_32FC1, (void*) im_ao);  
//     cv::Mat img_ao_mat = src(cv::Rect(cpt->imgpadding,cpt->imgpadding,cpt->width,cpt->height));
//     cv::Mat outimg;
//     img_ao_mat.convertTo(outimg, CV_8UC1);
//     cv::cvtColor(outimg, outimg, CV_GRAY2RGB);
//     cv::resize(outimg, outimg, cv::Size(), sc_fct_tmp, sc_fct_tmp, cv::INTER_NEAREST);
// 
//     for (int i = 0; i < nopatches; ++i)
//     {
//       // Show displacement vector
//       const Eigen::Vector2f pt_ret = pat[i]->GetPointPos();
//       
//       Eigen::Vector2f pta, ptb;
//       
//       cv::line(outimg, cv::Point( (pt_ref[i][0]+.5)*sc_fct_tmp, (pt_ref[i][1]+.5)*sc_fct_tmp ), cv::Point( (pt_ret[0]+.5)*sc_fct_tmp, (pt_ret[1]+.5)*sc_fct_tmp ), cv::Scalar(255*pat[i]->isConverged() ,255*(!pat[i]->isConverged()),0),  2);
//       
//       cv::line(outimg, cv::Point( (cpt->cx+.5)*sc_fct_tmp, (cpt->cy+.5)*sc_fct_tmp ), cv::Point( (cpt->cx+.5)*sc_fct_tmp, (cpt->cy+.5)*sc_fct_tmp ), cv::Scalar(0,0, 255),  2);
// 
//     }
// 
//     char str[200];
//     sprintf(str,"Iter: %i",cnt);
//     cv::putText(outimg, str, cv::Point2f(20,20), cv::FONT_HERSHEY_PLAIN, 1,  cv::Scalar(0,0,255,255), 2);
// 
//     cv::namedWindow( "Img_iter", cv::WINDOW_AUTOSIZE );
//     cv::imshow( "Img_iter", outimg);
//     
//     cv::waitKey(500);
//   }
// } 

void PatGridClass::InitializeFromCoarserOF(const float * flow_prev)
{
  #pragma omp parallel for schedule(dynamic,10)
  for (int ip = 0; ip < nopatches; ++ip)
  {
    int x = floor(pt_ref[ip][0] / 2); // better, but slower: use bil. interpolation here
    int y = floor(pt_ref[ip][1] / 2); 
    int i = y*(cpt->width/2) + x;
    
    #if (SELECTMODE==1)
    p_init[ip](0) = flow_prev[2*i  ]*2;
    p_init[ip](1) = flow_prev[2*i+1]*2;
    #else
    p_init[ip](0) = flow_prev[  i  ]*2;      
    #endif
  }
}

void PatGridClass::AggregateFlowDense(float *flowout) const
{
  float* we = new float[cpt->width * cpt->height];
  
  memset(flowout, 0, sizeof(float) * (op->nop * cpt->width * cpt->height) );
  memset(we,      0, sizeof(float) * (          cpt->width * cpt->height) );
  
  #ifdef USE_PARALLEL_ON_FLOWAGGR // Using this enables OpenMP on flow aggregation. This can lead to race conditions. Experimentally we found that the result degrades only marginally. However, for our experiments we did not enable this.
    #pragma omp parallel for schedule(static)  
  #endif
  for (int ip = 0; ip < nopatches; ++ip)
  {       
    
    if (pat[ip]->IsValid())
    {
      #if (SELECTMODE==1)
      const Eigen::Vector2f*            fl = pat[ip]->GetParam(); // flow displacement of this patch
      Eigen::Vector2f flnew;
      #else
      const Eigen::Matrix<float, 1, 1>* fl = pat[ip]->GetParam(); // horz. displacement of this patch
      Eigen::Matrix<float, 1, 1> flnew;
      #endif
      
      const float * pweight = pat[ip]->GetpWeightPtr(); // use image error as weight
      
      int lb = -op->p_samp_s/2;
      int ub = op->p_samp_s/2-1;
      
      for (int y = lb; y <= ub; ++y)
      {   
        for (int x = lb; x <= ub; ++x, ++pweight)
        {
          int yt = (y + pt_ref[ip][1]);
          int xt = (x + pt_ref[ip][0]);

          if (xt >= 0 && yt >= 0 && xt < cpt->width && yt < cpt->height)
          {
  
            int i = yt*cpt->width + xt;
              
            #if (SELECTCHANNEL==1 | SELECTCHANNEL==2)  // single channel/gradient image 
            float absw = 1.0f /  (float)(std::max(op->minerrval  ,*pweight));
            #else  // RGB image
            float absw = (float)(std::max(op->minerrval  ,*pweight)); ++pweight;
                  absw+= (float)(std::max(op->minerrval  ,*pweight)); ++pweight;
                  absw+= (float)(std::max(op->minerrval  ,*pweight));
            absw = 1.0f / absw;
            #endif
              
            flnew = (*fl) * absw;
            we[i] += absw;

            #if (SELECTMODE==1)
            flowout[2*i]   += flnew[0];
            flowout[2*i+1] += flnew[1];
            #else
            flowout[i] += flnew[0]; 
            #endif
          }
        }
      }
    }
  } 
  
  // if complementary (forward-backward merging) is given, integrate negative backward flow as well
  if (cg)
  {  
      Eigen::Vector4f wbil; // bilinear weight vector
      Eigen::Vector4i pos;
      
      #ifdef USE_PARALLEL_ON_FLOWAGGR
        #pragma omp parallel for schedule(static)  
      #endif    
      for (int ip = 0; ip < cg->nopatches; ++ip)
      {
        if (cg->pat[ip]->IsValid())
        {
          #if (SELECTMODE==1)
          const Eigen::Vector2f*            fl = (cg->pat[ip]->GetParam()); // flow displacement of this patch
          Eigen::Vector2f flnew;
          #else
          const Eigen::Matrix<float, 1, 1>* fl = (cg->pat[ip]->GetParam()); // horz. displacement of this patch
          Eigen::Matrix<float, 1, 1> flnew;
          #endif
        
          const Eigen::Vector2f rppos = cg->pat[ip]->GetPointPos(); // get patch position after optimization
          const float * pweight = cg->pat[ip]->GetpWeightPtr(); // use image error as weight
          
          Eigen::Vector2f resid;

          // compute bilinear weight vector
          pos[0] = ceil(rppos[0] +.00001); // make sure they are rounded up to natural number
          pos[1] = ceil(rppos[1] +.00001); // make sure they are rounded up to natural number
          pos[2] = floor(rppos[0]);
          pos[3] = floor(rppos[1]);

          resid[0] = rppos[0] - pos[2];
          resid[1] = rppos[1] - pos[3];
          wbil[0] = resid[0]*resid[1];
          wbil[1] = (1-resid[0])*resid[1];
          wbil[2] = resid[0]*(1-resid[1]);
          wbil[3] = (1-resid[0])*(1-resid[1]);

          int lb = -op->p_samp_s/2;
          int ub = op->p_samp_s/2-1;

          
          for (int y = lb; y <= ub; ++y)
          {   
            for (int x = lb; x <= ub; ++x, ++pweight)
            {
          
              int yt = y + pos[1]; 
              int xt = x + pos[0];
              if (xt >= 1 && yt >= 1 && xt < (cpt->width-1) && yt < (cpt->height-1))
              {
                
                #if (SELECTCHANNEL==1 | SELECTCHANNEL==2)  // single channel/gradient image
                float absw = 1.0f /  (float)(std::max(op->minerrval  ,*pweight));
                #else  // RGB
                float absw = (float)(std::max(op->minerrval  ,*pweight)); ++pweight;
                      absw+= (float)(std::max(op->minerrval  ,*pweight)); ++pweight;
                      absw+= (float)(std::max(op->minerrval  ,*pweight));
                absw = 1.0f / absw;
                #endif
              
              
                flnew = (*fl) * absw;
                
                int idxcc =  xt    +  yt   *cpt->width;
                int idxfc = (xt-1) +  yt   *cpt->width;
                int idxcf =  xt    + (yt-1)*cpt->width;
                int idxff = (xt-1) + (yt-1)*cpt->width;            
        
                we[idxcc] += wbil[0] * absw;
                we[idxfc] += wbil[1] * absw;
                we[idxcf] += wbil[2] * absw;
                we[idxff] += wbil[3] * absw;

                #if (SELECTMODE==1)
                flowout[2*idxcc  ] -= wbil[0] * flnew[0];   // use reversed flow 
                flowout[2*idxcc+1] -= wbil[0] * flnew[1];

                flowout[2*idxfc  ] -= wbil[1] * flnew[0];
                flowout[2*idxfc+1] -= wbil[1] * flnew[1];

                flowout[2*idxcf  ] -= wbil[2] * flnew[0];
                flowout[2*idxcf+1] -= wbil[2] * flnew[1];

                flowout[2*idxff  ] -= wbil[3] * flnew[0];
                flowout[2*idxff+1] -= wbil[3] * flnew[1];
                #else
                flowout[idxcc] -= wbil[0] * flnew[0]; // simple averaging of inverse horizontal displacement
                flowout[idxfc] -= wbil[1] * flnew[0];
                flowout[idxcf] -= wbil[2] * flnew[0];
                flowout[idxff] -= wbil[3] * flnew[0];
                #endif
              }
            }
          }
        }
      }
  } 
  
  #pragma omp parallel for schedule(static, 100)    
  // normalize each pixel by dividing displacement by aggregated weights from all patches
  for (int yi = 0; yi < cpt->height; ++yi)
  {
    for (int xi = 0; xi < cpt->width; ++xi)
    { 
      int i    = yi*cpt->width + xi;
      if (we[i]>0)
      {
        #if (SELECTMODE==1)        
        flowout[2*i  ] /= we[i];
        flowout[2*i+1] /= we[i];
        #else
        flowout[i] /= we[i];
        #endif
      }
    }
  }
  
  delete[] we;
}

}


