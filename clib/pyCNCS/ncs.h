/* 
 * C library for "Noise Correction Algorithm for sCMOS cameras".
 * 
 * Hazen 04/19
 */

#ifndef NCS_H

/*
 * This structure contains everything necessary to run NCS on a sub-region.
 */
typedef struct ncsSubRegion
{
  int r_size;                   /* Sub-region size, square. */
  int fft_size;                 /* Size of the FFT on the second axis. */
  
  double alpha;                 /* NCS alpha parameter value. */
  double normalization;         /* FFT normalization constant. */

  double *data;                 /* Image data (of size r_size x r_size). */
  double *gamma;                /* CMOS variance data (of size r_size x r_size). */
  double *otf_mask;             /* OTF mask (of size r_size x r_size). */
  double *t1;                   /* Temporary gradient storage. */
  double *t2;                   /* Temporary gradient storage. */
  
  lbfgsfloatval_t *u;           /* Current fit. */
  
  fftw_plan fft_forward;        /* FFT transform plan. */

  fftw_complex *u_fft;          /* FFT of current fit. */
  fftw_complex **u_fft_grad;    /* Fit gradient FFT storage. */

  lbfgs_parameter_t *param;     /* The parameters of the L-BFGS method. */
} ncsSubRegion;


/*
 * Functions.
 */
void ncsSRCalcLLGradient(ncsSubRegion *, double *);
double ncsSRCalcLogLikelihood(ncsSubRegion *);
void ncsSRCalcNCGradient(ncsSubRegion *, double *);
double ncsSRCalcNoiseContribution(ncsSubRegion *);
void ncsSRCleanup(ncsSubRegion *);
void ncsSRGetU(ncsSubRegion *, double *);
ncsSubRegion *ncsSRInitialize(int);
void ncsSRNewRegion(ncsSubRegion *, double *, double *);
void ncsSRSetOTFMask(ncsSubRegion *, double *);
void ncsSRSetU(ncsSubRegion *, double *);
int ncsSRSolve(ncsSubRegion *, double, int);
  
#endif
