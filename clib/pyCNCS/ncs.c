/* 
 * C library for "Noise Correction Algorithm for sCMOS cameras".
 *
 * Note: We assume that lbfgsfloat_val is a double.
 * 
 * Hazen 04/19
 */

/* Include */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <fftw3.h>
#include <lbfgs.h>

#include "ncs.h"


/*
 * ncsReduceNoise() 
 *
 * Run NCS noise reduction on an image.
 *
 * ncs_image - Pre-allocated storage for the NCS image.
 * image - Original image in e-.
 * gamma - CMOS variance in units of e-^2.
 * otf_mask - r_size x r_size array containing the OTF mask.
 * alpha - NCS alpha term. 
 * im_x - Image size (slow axis). 
 * im_y - Image size (fast axis).
 * r_size - otf_mask size.
 */
void ncsReduceNoise(double *ncs_image,
		    double *image,
		    double *gamma,
		    double *otf_mask,
		    double alpha,
		    int im_x,
		    int im_y,
		    int r_size)
{
  int i,j,k,l,m,n,o,p,q;
  int bx,by,res,s_size;
  ncsSubRegion *ncs_sr;

  printf("%d %d %d\n",im_x,im_y,r_size);
  
  /* Check OTF mask size. */
  if((r_size%2)!=0){
    printf("OTF mask size of %d is not divisible by 2!", r_size);
    return;
  }

  q = 0;
  s_size = r_size - 2;
  
  /* Initialization. */
  ncs_sr = ncsSRInitialize(r_size);
  ncsSRSetOTFMask(ncs_sr, otf_mask);

  /*
   * This is somewhat complicated by the goal of using a 1 pixel
   * pad around each sub region, and also using duplicate values
   * for sub regions that are on the edge of the image.
   */
  for(i=-1;i<(im_x+1);i+=s_size){
    if((i+r_size)>(im_x+1)){
      bx = im_x - r_size + 1;
    }
    else{
      bx = i;
    }
    
    for(j=-1;j<(im_y+1);j+=s_size){
      if((j+r_size)>(im_y+1)){
	by = im_y - r_size + 1;
      }
      else{
	by = j;
      }

      /* Copy sub-region. */
      for(k=0;k<r_size;k++){
	if((k + bx) < 0){
	  l = 0;
	}
	else if((k + bx)>=im_x){
	  l = (im_x - 1)*im_y;
	}
	else{
	  l = (k + bx)*im_y;
	}
	m = k*r_size;
	for(n=0;n<r_size;n++){
	  if((n + by) < 0){
	    o = l;
	  }
	  else if((n + by)>=im_y){
	    o = l + im_y - 1;
	  }
	  else{
	    o = l + n + by;
	  }
	  p = m + n;
	  ncs_sr->data[p] = image[o];
	  ncs_sr->gamma[p] = gamma[o];
	}
      }
      printf("%d %d\n",bx,by);
      
      /* Solve. */
      res = ncsSRSolve(ncs_sr, alpha, 0);
      if(res!=0){
	printf("NCS solver failed on region %d %d with code %d!\n",bx,by,res);
      }
      
      /* Copy results. */
      for(k=1;k<(r_size-1);k++){
	/*
	if (bx == (im_x - r_size)){
	  l = (k + bx + 1)*im_y;
	}
	else{
	  l = (k + bx)*im_y;
	}
	*/
	l = (k + bx)*im_y;
	m = k*r_size;
	for(n=1;n<(r_size-1);n++){
	  /*
	  if(by == (im_y - r_size)){
	    o[ = l + n + by + 1;
	  }[
	  else{
	    o = l + n + by;
	  }
	  */
	  o = l + n + by;
	  p = m + n;
	  //ncs_image[o] = ncs_sr->u[p];
	  printf("1. %d %d\n",o,p);
	  ncs_image[o] = q;
	}
	printf("\n");
      }
      q += 1;
      
      /* 
       * This keeps us from analyzing the outer edge twice, which
       * can happen depending the values of s_size and im_y.
       */
      if(by == im_y - r_size + 1){
	break;
      }
    }
    if(bx == im_x - r_size + 1){
      break;
    }
  }
  
  /* Clean up. */
  ncsSRCleanup(ncs_sr);
}


/* 
 * ncsSRCalcLLGradient()
 * 
 * Calculate the gradient of the log-likelihood with current u, data and gamma.
 *
 * ncs_sr - Pointer to a ncsSubRegion structure.
 * gradient - Pointer to an array of doubles
 */
void ncsSRCalcLLGradient(ncsSubRegion *ncs_sr, double *gradient)
{
  int i,size;
  double t1,t2;

  size = ncs_sr->r_size;
  for(i=0;i<(size*size);i++){
    t1 = ncs_sr->data[i] + ncs_sr->gamma[i];
    t2 = ncs_sr->u[i] + ncs_sr->gamma[i];
    gradient[i] = 1 - t1/t2;
  }
}


/* 
 * ncsSRCalcLogLikelihood()
 * 
 * Calculate the current log-likelihood with current u, data and gamma.
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
 * ncsSRCalcNCGradient()
 * 
 * Calculate the gradient of the current noise contribution. The expectation
 * is that will be called after ncsSRCalcNoiseContribution() which will 
 * calculate the FFT of the current fit.
 *
 * ncs_sr - Pointer to a ncsSubRegion structure.
 */
void ncsSRCalcNCGradient(ncsSubRegion *ncs_sr, double *gradient)
{
  int i,j,k,l,m,size,fft_size;
  double sum,t1,t2;
  fftw_complex *ft,*ft_g;

  size = ncs_sr->r_size;
  fft_size = ncs_sr->fft_size;
  ft = ncs_sr->u_fft;
  
  for(i=0;i<(size*size);i++){
    ft_g = ncs_sr->u_fft_grad[i];
    sum = 0.0;
    for(j=0;j<size;j++){
      for(k=0;k<size;k++){
	l = j*size+k;

	/* FFT has a different size than the OTF mask. */
	if(k>=fft_size){
	  if(j==0){
	    m = size-k;
	  }
	  else{
	    m = (size-j+1)*fft_size - k + (size - fft_size);
	  }
	}
	else{
	  m = j*fft_size + k;  
	}

	t1 = ft_g[m][0]*ft[m][0] + ft_g[m][1]*ft[m][1];
	t2 = ncs_sr->otf_mask[l];
	sum += 2.0*t1*t2*t2;
      }
    }
    gradient[i] = sum;
  }
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
  int i;
  
  free(ncs_sr->data);
  free(ncs_sr->gamma);
  free(ncs_sr->otf_mask);
  free(ncs_sr->t1);
  free(ncs_sr->t2);
  
  lbfgs_free(ncs_sr->u);
  
  fftw_destroy_plan(ncs_sr->fft_forward);
  
  fftw_free(ncs_sr->u_fft);

  for(i=0;i<(ncs_sr->r_size*ncs_sr->r_size);i++){
    fftw_free(ncs_sr->u_fft_grad[i]);
  }
  free(ncs_sr->u_fft_grad);

  free(ncs_sr->param);
  
  free(ncs_sr);
}


/*
 * ncsSREvaluate()
 *
 * Callback for solver updates, used by L-BFGS method.
 */
static lbfgsfloatval_t ncsSREvaluate(void *instance,
				     const lbfgsfloatval_t *x,
				     lbfgsfloatval_t *g,
				     const int n,
				     const lbfgsfloatval_t step)
{
  int i,size;
  lbfgsfloatval_t fx;
  ncsSubRegion *ncs_sr;

  ncs_sr = (ncsSubRegion *)instance;
  size = ncs_sr->r_size;

  /*
   * Calculate current 'cost'.
   */
  fx = ncsSRCalcLogLikelihood(ncs_sr);
  fx += ncs_sr->alpha*ncsSRCalcNoiseContribution(ncs_sr);

  /*
   * Calculate cost gradient.
   */
  ncsSRCalcLLGradient(ncs_sr, ncs_sr->t1);
  ncsSRCalcNCGradient(ncs_sr, ncs_sr->t2);
  for(i=0;i<(size*size);i++){
    g[i] = ncs_sr->t1[i] + ncs_sr->alpha*ncs_sr->t2[i];
  }
  
  return fx;
}


/*
 * ncsSRGetU()
 *
 * Get the current u vector, usually would call this after ncsSRSolve().
 *
 * ncs_sr - Pointer to ncsSubRegion structure.
 * u - Pre-allocated storage for the u vector.
 */
void ncsSRGetU(ncsSubRegion *ncs_sr, double *u)
{
  int i,size;

  size = ncs_sr->r_size;
  for(i=0;i<(size*size);i++){
    u[i] = ncs_sr->u[i];
  }
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
  int i,j,fft_size;
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
  ncs_sr->t1 = (double *)malloc(sizeof(double)*r_size*r_size);
  ncs_sr->t2 = (double *)malloc(sizeof(double)*r_size*r_size);
  
  ncs_sr->u = lbfgs_malloc(r_size*r_size);
  
  for(i=0;i<(r_size*r_size);i++){
    ncs_sr->data[i] = 0.0;
    ncs_sr->gamma[i] = 0.0;
    ncs_sr->otf_mask[i] = 0.0;
    ncs_sr->t1[i] = 0.0;
    ncs_sr->t2[i] = 0.0;
 
    ncs_sr->u[i] = 0.0;
  }

  fft_size = r_size/2 + 1;
  ncs_sr->fft_size = fft_size; 
  ncs_sr->normalization = 1.0/((double)r_size);

  ncs_sr->u_fft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*r_size*fft_size);
  ncs_sr->fft_forward = fftw_plan_dft_r2c_2d(r_size, r_size, (double *)ncs_sr->u, ncs_sr->u_fft, FFTW_ESTIMATE);

  /* 
   * This is an optimization for calculating the noise contribution gradient. For
   * the gradient calculation we need the FFT of arrays that are all zero except
   * for a 1.0 at a single point. To save effort we calculate all these FFTs now.
   */
  ncs_sr->u_fft_grad = (fftw_complex **)malloc(sizeof(fftw_complex *)*r_size*r_size);
  for(i=0;i<(r_size*r_size);i++){
    ncs_sr->u_fft_grad[i] = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)*r_size*fft_size);
			     
    ncs_sr->u[i] = 1.0;
    fftw_execute(ncs_sr->fft_forward);
    for(j=0;j<(r_size*fft_size);j++){
      ncs_sr->u_fft_grad[i][j][0] = ncs_sr->u_fft[j][0]*ncs_sr->normalization;
      ncs_sr->u_fft_grad[i][j][1] = ncs_sr->u_fft[j][1]*ncs_sr->normalization;
    }
    ncs_sr->u[i] = 0.0;
  }

  /* L-BFGS parameter initialization. */
  ncs_sr->param = (lbfgs_parameter_t *)malloc(sizeof(lbfgs_parameter_t));
  lbfgs_parameter_init(ncs_sr->param);
  
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
void ncsSRNewRegion(ncsSubRegion *ncs_sr, double *image, double *gamma)
{
  int i,size;
  
  size = ncs_sr->r_size;
  for(i=0;i<(size*size);i++){
    ncs_sr->data[i] = image[i];
    ncs_sr->gamma[i] = gamma[i];
  }
}


/*
 * ncsSRProgress()
 *
 * Callback for reporting solver progress, used by L-BFGS method.
 */
static int ncsSRProgress(void *instance,
			 const lbfgsfloatval_t *x,
			 const lbfgsfloatval_t *g,
			 const lbfgsfloatval_t fx,
			 const lbfgsfloatval_t xnorm,
			 const lbfgsfloatval_t gnorm,
			 const lbfgsfloatval_t step,
			 int n,
			 int k,
			 int ls)
{
    printf("Iteration %d:\n", k);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
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


/*
 * ncsSRSolve()
 *
 * Solve for optimal u using the L-BFGS algorithm.
 *
 * ncs_sr - Pointer to a ncsSubRegion structure.
 */
int ncsSRSolve(ncsSubRegion *ncs_sr, double alpha, int verbose)
{
  int i,ret,size;
  lbfgsfloatval_t fx;
  
  ncs_sr->alpha = alpha;

  size = ncs_sr->r_size;

  /* Starting values are the current image. */
  for(i=0;i<(size*size);i++){
    ncs_sr->u[i] = ncs_sr->data[i];
  }

  if (verbose){
    ret = lbfgs(size*size, ncs_sr->u, &fx, ncsSREvaluate, ncsSRProgress, (void *)ncs_sr, ncs_sr->param);
  }
  else{
    ret = lbfgs(size*size, ncs_sr->u, &fx, ncsSREvaluate, NULL, (void *)ncs_sr, ncs_sr->param);
  }

  return ret;
}
