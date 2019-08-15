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
  double *g;                    /* Temporary gradient storage. */
  double *gamma;                /* CMOS variance data (of size r_size x r_size). */
  double *otf_mask_sqr;         /* OTF mask squared (of size r_size x r_size). */
  double *t1;                   /* Temporary storage (used in gradient calculation). */
  double *t2;                   /* Temporary storage (used in gradient calculation). */

  lbfgsfloatval_t *u;           /* Current fit. */
  
  fftw_plan fft_backward;       /* IFFT transform plan. */
  fftw_plan fft_forward;        /* FFT transform plan. */

  fftw_complex *g_fft;          /* FFT of NC gradient. */
  fftw_complex *u_fft;          /* FFT of current fit. */

  lbfgs_parameter_t *param;     /* The parameters of the L-BFGS method. */
} ncsSubRegion;


/*
 * Functions.
 */
void ncsReduceNoise(double *, double *, double *, double *, double, int, int, int);
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
