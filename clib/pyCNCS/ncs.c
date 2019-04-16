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
 * ncsSRCalcLogLikelihood()
 * 
 * Calculate the current log-likelihood given u, data and gamma.
 *
 * ncs_sr - Pointer to a ncsSubRegion structure.
 */
double ncsSRCalcLogLikelihood(ncsSubRegion *ncs_sr)
{
  int i,size;
  double sum,t1,t2;

  size = ncs_sr->r_size;
  sum = 0.0;
  for(i=0;i<(size*size);i++){
    t1 = ncs_sr->data[i] + ncs_sr->gamma[i];
    t2 = log(ncs_sr->u[i] + ncs_sr->gamma[i]);
    sum += ncs_sr->u[i] - t1*t2;
  }
  
  return sum;
}


/* 
 * ncsSRCalcNoiseContribution()
 * 
 * Calculate the current noise contribution.
 *
 * ncs_sr - Pointer to a ncsSubRegion structure.
 */
double ncsSRCalcNoiseContribution(ncsSubRegion *ncs_sr)
{
  int i,j,k,l,size,fft_size;
  double sum,t1,t2;

  size = ncs_sr->r_size;
  fft_size = ncs_sr->fft_size;

  /* Compute FFT of the current estimate. */
  fftw_execute(ncs_sr->fft_forward);

  /* Normalize FFT. */
  for(i=0;i<(size*fft_size);i++){
    ncs_sr->u_fft[i][0] = ncs_sr->u_fft[i][0]*ncs_sr->normalization;
    ncs_sr->u_fft[i][1] = ncs_sr->u_fft[i][1]*ncs_sr->normalization;
  }

  /*
   * FIXME: It seems like there should be some symmetries here that we could
   *        take advantage of to simplify the math and the indexing. The OTF
   *        mask is expected to be symmetric.
   */
  sum = 0.0;
  for(i=0;i<size;i++){
    for(j=0;j<size;j++){
      k = i*size+j;

      /* FFT has a different size than the OTF mask. */
      if(j>=fft_size){
	if(i==0){
	  l = size-j;
	}
	else{
	  l = (size-i+1)*fft_size - j + (size - fft_size);
	}
      }
      else{
	l = i*fft_size + j;  
      }
      
      t1 = ncs_sr->u_fft[l][0]*ncs_sr->u_fft[l][0] + ncs_sr->u_fft[l][1]*ncs_sr->u_fft[l][1];
      t2 = ncs_sr->otf_mask[k];
      sum += t1*t2*t2;
    }
  }

  return sum;
}


/*
 * ncsSRCleanup()
 *
 * ncs_sr - Pointer to ncsSubRegion structure.
 */
void ncsSRCleanup(ncsSubRegion *ncs_sr)
{
  free(ncs_sr->data);
  free(ncs_sr->gamma);
  free(ncs_sr->otf_mask);
  free(ncs_sr->u);
  
  fftw_destroy_plan(ncs_sr->fft_forward);
  
  fftw_free(ncs_sr->u_fft);

  free(ncs_sr);
}


/*
 * ncsSRInitialize()
 *
 * Set things up for NCS for a sub-region.
 *
 * r_size - The size of the square sub-region, must be divisible by 2. Typically
 *          this is a power of 2.
 */
ncsSubRegion *ncsSRInitialize(int r_size)
{
  int i;
  ncsSubRegion *ncs_sr;

  /* Check that r_size is a divisible by 2. */
  if ((r_size%2)!=0){
    printf("ROI size must be divisible by 2!\n");
    return NULL;
  }
  
  ncs_sr = (ncsSubRegion *)malloc(sizeof(ncsSubRegion));
  
  ncs_sr->r_size = r_size;
  ncs_sr->alpha = 0.0;

  ncs_sr->data = (double *)malloc(sizeof(double)*r_size*r_size);
  ncs_sr->gamma = (double *)malloc(sizeof(double)*r_size*r_size);
  ncs_sr->otf_mask = (double *)malloc(sizeof(double)*r_size*r_size);
  ncs_sr->u = (double *)malloc(sizeof(double)*r_size*r_size);

  for(i=0;i<(r_size*r_size);i++){
    ncs_sr->data[i] = 0.0;
    ncs_sr->gamma[i] = 0.0;
    ncs_sr->otf_mask[i] = 0.0;
    ncs_sr->u[i] = 0.0;
  }

  ncs_sr->fft_size = r_size/2 + 1; 
  ncs_sr->normalization = 1.0/((double)r_size);

  ncs_sr->u_fft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*r_size*ncs_sr->fft_size);
  ncs_sr->fft_forward = fftw_plan_dft_r2c_2d(r_size, r_size, ncs_sr->u, ncs_sr->u_fft, FFTW_ESTIMATE);

  return ncs_sr;
}


/*
 * ncsSRNewRegion()
 *
 * Start analysis of a new sub-region.
 *
 * ncs_sr - Pointer to ncsSubRegion structure.
 * image - The image sub-region.
 * gamma - The CMOS variance in the sub-region.
 * alpha - Alpha parameter to use when solving.
 */
void ncsSRNewRegion(ncsSubRegion *ncs_sr, double *image, double *gamma, double alpha)
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
 * ncsSRSetOTFMask()
 *
 * Initialize / change OTF mask.
 *
 * ncs_sr - Pointer to ncsSubRegion structure.
 * otf - The microscopes OTF.
 */
void ncsSRSetOTFMask(ncsSubRegion *ncs_sr, double *otf_mask)
{
  int i,size;

  size = ncs_sr->r_size;
  for(i=0;i<(size*size);i++){
    ncs_sr->otf_mask[i] = otf_mask[i];
  }
}


/*
 * ncsSRSetU()
 *
 * Set the u vector, this is primarily used for testing.
 *
 * ncs_sr - Pointer to ncsSubRegion structure.
 * u - The new u vector.
 */
void ncsSRSetU(ncsSubRegion *ncs_sr, double *u)
{
  int i,size;

  size = ncs_sr->r_size;
  for(i=0;i<(size*size);i++){
    ncs_sr->u[i] = u[i];
  }
}
