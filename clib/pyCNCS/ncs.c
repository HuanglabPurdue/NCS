/* 
 * C library for "Noise Correction Algorithm for sCMOS cameras".
 * 
 * Hazen 04/19
 */

/* Include */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <fftw3.h>

#include "ncs.h"


/*
 * ncsSRCleanup()
 *
 * ncs_sr - Pointer to ncsSubRegion structure.
 */
void ncsSRCleanup(ncsSubRegion *ncs_sr)
{
  free(ncs_sr->data);
  free(ncs_sr->gamma);
  free(ncs_sr->otf);
  
  fftw_destroy_plan(ncs_sr->fft_forward);
  
  fftw_free(ncs_sr->u_fft);

  free(ncs_sr);
}


/*
 * ncsSRInitialize()
 *
 * Set things up for NCS for a sub-region.
 *
 * r_size - The size of the square sub-region, usually a power of 2 like 16.
 */
ncsSubRegion *ncsSRInitialize(int r_size)
{
  int i;
  ncsSubRegion *ncs_sr;

  ncs_sr = (ncsSubRegion *)malloc(sizeof(ncsSubRegion));
  
  ncs_sr->r_size = r_size;
  ncs_sr->alpha = 0.0;

  ncs_sr->data = (double *)malloc(sizeof(double)*r_size*r_size);
  ncs_sr->gamma = (double *)malloc(sizeof(double)*r_size*r_size);
  ncs_sr->otf = (double *)malloc(sizeof(double)*r_size*r_size);
  ncs_sr->u = (double *)malloc(sizeof(double)*r_size*r_size);

  for(i=0;i<(r_size*r_size);i++){
    ncs_sr->data[i] = 0.0;
    ncs_sr->gamma[i] = 0.0;
    ncs_sr->otf[i] = 0.0;
    ncs_sr->u[i] = 0.0;
  }
  
  ncs_sr->fft_size = r_size * (r_size/2 + 1);
  ncs_sr->normalization = 1.0/((double)(r_size * r_size));

  ncs_sr->u_fft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*ncs_sr->fft_size);
  ncs_sr->fft_forward = fftw_plan_dft_r2c_2d(r_size, r_size, ncs_sr->u, ncs_sr->u_fft, FFTW_ESTIMATE);

  return ncs_sr;
}


/*
 * ncsSRNewData()
 *
 * ncs_sr - Pointer to ncsSubRegion structure.
 * image - The image sub-region.
 * gamma - The CMOS variance in the sub-region.
 * alpha - Alpha parameter to use when solving.
 */
void ncsSRNewData(ncsSubRegion *ncs_sr, double *image, double *gamma, double alpha)
{
  int i,size;

  ncs_sr->alpha = alpha;
  
  size = ncs_sr->r_size;
  for(i=0;i<(size*size);i++){
    ncs_sr->data[i] = image[i];
    ncs_sr->gamma[i] = gamma[i];
    ncs_sr->u[i] = image[i];
  }
}


/*
 * ncsSRSetOTF()
 *
 * ncs_sr - Pointer to ncsSubRegion structure.
 * otf - The micrscopes OTF.
 */
void ncsSRSetOTF(ncsSubRegion *ncs_sr, double *otf)
{
  int i,size;

  size = ncs_sr->r_size;
  for(i=0;i<(size*size);i++){
    ncs_sr->otf[i] = otf[i];
  }
}
