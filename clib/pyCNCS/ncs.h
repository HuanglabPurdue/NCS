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
  double *u;                    /* Current fit. */
  
  fftw_plan fft_forward;        /* FFT transform plan. */

  fftw_complex *u_fft;          /* FFT of current fit. */
  
} ncsSubRegion;


/*
 * Functions.
 */

void ncsSRCalcLLGradient(ncsSubRegion *, double *);
double ncsSRCalcLogLikelihood(ncsSubRegion *);
double ncsSRCalcNoiseContribution(ncsSubRegion *);
void ncsSRCleanup(ncsSubRegion *);
ncsSubRegion *ncsSRInitialize(int);
void ncsSRNewRegion(ncsSubRegion *, double *, double *, double);
void ncsSRSetOTFMask(ncsSubRegion *, double *);
void ncsSRSetU(ncsSubRegion *, double *);

#endif
