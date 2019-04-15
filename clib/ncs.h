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
  int fft_size;                 /* Size of the FFT. */
  int r_size;                   /* Sub-region size, square. */
  
  double alpha;                 /* NCS alpha parameter value. */
  double normalization;         /* FFT normalization constant. */

  double *data;                 /* Image data (of size r_size x r_size). */
  double *gamma;                /* CMOS variance data (of size r_size x r_size). */
  double *otf;                  /* OTF (of size r_size x r_size). */
  double *u;                    /* Current fit. */
  
  fftw_plan fft_forward;        /* FFT transform plan. */

  fftw_complex *u_fft;          /* FFT of current fit. */
  
} ncsSubRegion;


/*
 * Functions.
 */

void ncsSRCleanup(ncsSubRegion *);
ncsSubRegion *ncsSRInitialize(int);
void ncsSRNewData(ncsSubRegion *, double *, double *, double);
void ncsSRSetOTF(ncsSubRegion *, double *);

#endif
