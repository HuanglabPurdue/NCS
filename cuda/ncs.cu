/*
 * CUDA kernel code for NCS.
 *
 * Important note!! This will only work correctly with 16 work items per work group!!
 * Why 16? This matches the 1D size of the 2D FFT.
 *
 * Hazen 08/19
 */

/*
 * License for the Limited memory BFGS (L-BFGS) solver code.
 *
 * Limited memory BFGS (L-BFGS).
 *
 * Copyright (c) 1990, Jorge Nocedal
 * Copyright (c) 2007-2010 Naoaki Okazaki
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <helper_math.h>

/* Threshold for handling negative values in the fit. */
#define FITMIN 1.0e-6f

/*
 * The problem size is (16*16)/4 or 256/4.
 *
 * This matches the size of the FFT so changing these would involve writing
 * a different size FFT, etc..
 */
#define ASIZE 16
#define PSIZE 64

/* L-BFGS solver parameters. */

#define C_1 1.0e-4f        /* Armijo rule/condition scaling value. */
#define EPSILON 1.0e-4f    /* Stopping point. */
#define M 8                /* Number of history points saved. Must be a power of 2. */
#define MAXITERS 200       /* Maximum number of iterations. */
#define MIN_STEP 1.0e-6f   /* Minimum step size. */
#define STEPM 0.5          /* Step size multiplier. */

/* Error status codes. */
#define SUCCESS 0
#define UNSTARTED -1
#define REACHED_MAXITERS -2
#define INCREASING_GRADIENT -3
#define MINIMUM_STEP -4
#define REACHED_MAXPRECISION -5

/****************
 * FFT functions.
 ****************/

// 4 point complex FFT
__device__ void fft4(float4 x_r, float4 x_c, float4 *y_r, float4 *y_c)
{
    float t1_r = x_r.x + x_r.z;
    float t1_c = x_c.x + x_c.z;
    
    float t2_r = x_r.x - x_r.z;
    float t2_c = x_c.x - x_c.z;

    float t3_r = x_r.y + x_r.w;
    float t3_c = x_c.y + x_c.w;

    float t4_r = x_r.y - x_r.w;
    float t4_c = x_c.y - x_c.w;
 
    y_r[0].x = t1_r + t3_r;
    y_r[0].y = t2_r + t4_c;
    y_r[0].z = t1_r - t3_r;
    y_r[0].w = t2_r - t4_c;

    y_c[0].x = t1_c + t3_c;
    y_c[0].y = t2_c - t4_r;
    y_c[0].z = t1_c - t3_c;
    y_c[0].w = t2_c + t4_r;
}


// 4 point complex IFFT
__device__ void ifft4(float4 x_r, float4 x_c, float4 *y_r, float4 *y_c)
{
    float t1_r = x_r.x + x_r.z;
    float t1_c = x_c.x + x_c.z;
    
    float t2_r = x_r.x - x_r.z;
    float t2_c = x_c.x - x_c.z;

    float t3_r = x_r.y + x_r.w;
    float t3_c = x_c.y + x_c.w;

    float t4_r = x_r.y - x_r.w;
    float t4_c = x_c.y - x_c.w;
 
    y_r[0].x = t1_r + t3_r;
    y_r[0].y = t2_r - t4_c;
    y_r[0].z = t1_r - t3_r;
    y_r[0].w = t2_r + t4_c;

    y_c[0].x = t1_c + t3_c;
    y_c[0].y = t2_c + t4_r;
    y_c[0].z = t1_c - t3_c;
    y_c[0].w = t2_c - t4_r;
    
    y_r[0] = y_r[0]*0.25f;
    y_c[0] = y_c[0]*0.25f;
}


// 8 point complex FFT.
__device__ void fft8(float4 *x_r, float4 *x_c, float4 *y_r, float4 *y_c)
{
    float4 t_r;
    float4 t_c;
    float4 f41_r;
    float4 f41_c;
    float4 f42_r;
    float4 f42_c;
     
    // 4 point FFT.
    t_r = make_float4(x_r[0].x, x_r[0].z, x_r[1].x, x_r[1].z);
    t_c = make_float4(x_c[0].x, x_c[0].z, x_c[1].x, x_c[1].z);    
    fft4(t_r, t_c, &f41_r, &f41_c);

    t_r = make_float4(x_r[0].y, x_r[0].w, x_r[1].y, x_r[1].w);
    t_c = make_float4(x_c[0].y, x_c[0].w, x_c[1].y, x_c[1].w);
    fft4(t_r, t_c, &f42_r, &f42_c);
    
    // Shift and add.
    float4 r1 = make_float4(1.0f, 7.07106781e-01f, 0.0f, -7.07106781e-01f);
    float4 c1 = make_float4(0.0f, 7.07106781e-01f, 1.0f,  7.07106781e-01f);
    y_r[0] = f41_r + f42_r * r1 + f42_c * c1;
    y_c[0] = f41_c + f42_c * r1 - f42_r * c1;
    y_r[1] = f41_r - f42_r * r1 - f42_c * c1;
    y_c[1] = f41_c - f42_c * r1 + f42_r * c1;    
}


// 8 point complex IFFT.
__device__ void ifft8(float4 *x_r, float4 *x_c, float4 *y_r, float4 *y_c)
{
    float4 t_r;
    float4 t_c;
    float4 f41_r;
    float4 f41_c;
    float4 f42_r;
    float4 f42_c;
     
    // 4 point IFFT.
    t_r = make_float4(x_r[0].x, x_r[0].z, x_r[1].x, x_r[1].z);
    t_c = make_float4(x_c[0].x, x_c[0].z, x_c[1].x, x_c[1].z);    
    ifft4(t_r, t_c, &f41_r, &f41_c);

    t_r = make_float4(x_r[0].y, x_r[0].w, x_r[1].y, x_r[1].w);
    t_c = make_float4(x_c[0].y, x_c[0].w, x_c[1].y, x_c[1].w);
    ifft4(t_r, t_c, &f42_r, &f42_c);
    
    // Shift and add.
    float4 r1 = make_float4(1.0f,  7.07106781e-01f,  0.0f, -7.07106781e-01f);
    float4 c1 = make_float4(0.0f, -7.07106781e-01f, -1.0f, -7.07106781e-01f);
    y_r[0] = f41_r + f42_r * r1 + f42_c * c1;
    y_c[0] = f41_c + f42_c * r1 - f42_r * c1;
    y_r[1] = f41_r - f42_r * r1 - f42_c * c1;
    y_c[1] = f41_c - f42_c * r1 + f42_r * c1;
    
    y_r[0] = y_r[0]*0.5f;
    y_c[0] = y_c[0]*0.5f;
    y_r[1] = y_r[1]*0.5f;
    y_c[1] = y_c[1]*0.5f;
}


// 16 point complex FFT.
__device__ void fft16(float4 *x_r, float4 *x_c, float4 *y_r, float4 *y_c)
{
    float4 t_r[2];
    float4 t_c[2];
    float4 f41_r[2];
    float4 f41_c[2];
    float4 f42_r[2];
    float4 f42_c[2];
     
    // 8 point FFT.
    t_r[0] = make_float4(x_r[0].x, x_r[0].z, x_r[1].x, x_r[1].z);
    t_c[0] = make_float4(x_c[0].x, x_c[0].z, x_c[1].x, x_c[1].z);    
    t_r[1] = make_float4(x_r[2].x, x_r[2].z, x_r[3].x, x_r[3].z);
    t_c[1] = make_float4(x_c[2].x, x_c[2].z, x_c[3].x, x_c[3].z);    
    fft8(t_r, t_c, f41_r, f41_c);

    t_r[0] = make_float4(x_r[0].y, x_r[0].w, x_r[1].y, x_r[1].w);
    t_c[0] = make_float4(x_c[0].y, x_c[0].w, x_c[1].y, x_c[1].w);
    t_r[1] = make_float4(x_r[2].y, x_r[2].w, x_r[3].y, x_r[3].w);
    t_c[1] = make_float4(x_c[2].y, x_c[2].w, x_c[3].y, x_c[3].w);
    fft8(t_r, t_c, f42_r, f42_c);
        
    // Shift and add.
    float4 r1 = make_float4(1.0f, 9.23879533e-01f, 7.07106781e-01f, 3.82683432e-01f);
    float4 c1 = make_float4(0.0f, 3.82683432e-01f, 7.07106781e-01f, 9.23879533e-01f);
    y_r[0] = f41_r[0] + f42_r[0] * r1 + f42_c[0] * c1;
    y_c[0] = f41_c[0] + f42_c[0] * r1 - f42_r[0] * c1;    
    y_r[2] = f41_r[0] - f42_r[0] * r1 - f42_c[0] * c1;
    y_c[2] = f41_c[0] - f42_c[0] * r1 + f42_r[0] * c1;    
    
    r1 = make_float4(0.0f, -3.82683432e-01f, -7.07106781e-01f, -9.23879533e-01f);
    c1 = make_float4(1.0f,  9.23879533e-01f,  7.07106781e-01f,  3.82683432e-01f);
    y_r[1] = f41_r[1] + f42_r[1] * r1 + f42_c[1] * c1;
    y_c[1] = f41_c[1] + f42_c[1] * r1 - f42_r[1] * c1;    
    y_r[3] = f41_r[1] - f42_r[1] * r1 - f42_c[1] * c1;
    y_c[3] = f41_c[1] - f42_c[1] * r1 + f42_r[1] * c1;    
}


// 16 point complex FFT (__local variable version).
__device__ void fft16_lvar(float4 *x_r, float4 *x_c, float4 *y_r, float4 *y_c)
{
    float4 t_r[2];
    float4 t_c[2];
    float4 f41_r[2];
    float4 f41_c[2];
    float4 f42_r[2];
    float4 f42_c[2];
     
    // 8 point FFT.
    t_r[0] = make_float4(x_r[0].x, x_r[0].z, x_r[1].x, x_r[1].z);
    t_c[0] = make_float4(x_c[0].x, x_c[0].z, x_c[1].x, x_c[1].z);    
    t_r[1] = make_float4(x_r[2].x, x_r[2].z, x_r[3].x, x_r[3].z);
    t_c[1] = make_float4(x_c[2].x, x_c[2].z, x_c[3].x, x_c[3].z);    
    fft8(t_r, t_c, f41_r, f41_c);

    t_r[0] = make_float4(x_r[0].y, x_r[0].w, x_r[1].y, x_r[1].w);
    t_c[0] = make_float4(x_c[0].y, x_c[0].w, x_c[1].y, x_c[1].w);
    t_r[1] = make_float4(x_r[2].y, x_r[2].w, x_r[3].y, x_r[3].w);
    t_c[1] = make_float4(x_c[2].y, x_c[2].w, x_c[3].y, x_c[3].w);
    fft8(t_r, t_c, f42_r, f42_c);
        
    // Shift and add.
    float4 r1 = make_float4(1.0f, 9.23879533e-01f, 7.07106781e-01f, 3.82683432e-01f);
    float4 c1 = make_float4(0.0f, 3.82683432e-01f, 7.07106781e-01f, 9.23879533e-01f);
    y_r[0] = f41_r[0] + f42_r[0] * r1 + f42_c[0] * c1;
    y_c[0] = f41_c[0] + f42_c[0] * r1 - f42_r[0] * c1;    
    y_r[2] = f41_r[0] - f42_r[0] * r1 - f42_c[0] * c1;
    y_c[2] = f41_c[0] - f42_c[0] * r1 + f42_r[0] * c1;    

    r1 = make_float4(0.0f, -3.82683432e-01f, -7.07106781e-01f, -9.23879533e-01f);
    c1 = make_float4(1.0f,  9.23879533e-01f,  7.07106781e-01f,  3.82683432e-01f);
    y_r[1] = f41_r[1] + f42_r[1] * r1 + f42_c[1] * c1;
    y_c[1] = f41_c[1] + f42_c[1] * r1 - f42_r[1] * c1;    
    y_r[3] = f41_r[1] - f42_r[1] * r1 - f42_c[1] * c1;
    y_c[3] = f41_c[1] - f42_c[1] * r1 + f42_r[1] * c1;    
}


// 16 point complex IFFT.
__device__ void ifft16(float4 *x_r, float4 *x_c, float4 *y_r, float4 *y_c)
{
    float4 t_r[2];
    float4 t_c[2];
    float4 f41_r[2];
    float4 f41_c[2];
    float4 f42_r[2];
    float4 f42_c[2];
     
    // 8 point IFFT.
    t_r[0] = make_float4(x_r[0].x, x_r[0].z, x_r[1].x, x_r[1].z);
    t_c[0] = make_float4(x_c[0].x, x_c[0].z, x_c[1].x, x_c[1].z);    
    t_r[1] = make_float4(x_r[2].x, x_r[2].z, x_r[3].x, x_r[3].z);
    t_c[1] = make_float4(x_c[2].x, x_c[2].z, x_c[3].x, x_c[3].z);    
    ifft8(t_r, t_c, f41_r, f41_c);

    t_r[0] = make_float4(x_r[0].y, x_r[0].w, x_r[1].y, x_r[1].w);
    t_c[0] = make_float4(x_c[0].y, x_c[0].w, x_c[1].y, x_c[1].w);
    t_r[1] = make_float4(x_r[2].y, x_r[2].w, x_r[3].y, x_r[3].w);
    t_c[1] = make_float4(x_c[2].y, x_c[2].w, x_c[3].y, x_c[3].w);
    ifft8(t_r, t_c, f42_r, f42_c);
        
    // Shift and add.
    float4 r1 = make_float4(1.0f,  9.23879533e-01f,  7.07106781e-01f,  3.82683432e-01f);
    float4 c1 = make_float4(0.0f, -3.82683432e-01f, -7.07106781e-01f, -9.23879533e-01f);
    y_r[0] = f41_r[0] + f42_r[0] * r1 + f42_c[0] * c1;
    y_c[0] = f41_c[0] + f42_c[0] * r1 - f42_r[0] * c1;    
    y_r[2] = f41_r[0] - f42_r[0] * r1 - f42_c[0] * c1;
    y_c[2] = f41_c[0] - f42_c[0] * r1 + f42_r[0] * c1;    
    
    r1 = make_float4( 0.0f, -3.82683432e-01f, -7.07106781e-01f, -9.23879533e-01f);
    c1 = make_float4(-1.0f, -9.23879533e-01f, -7.07106781e-01f, -3.82683432e-01f);
    y_r[1] = f41_r[1] + f42_r[1] * r1 + f42_c[1] * c1;
    y_c[1] = f41_c[1] + f42_c[1] * r1 - f42_r[1] * c1;    
    y_r[3] = f41_r[1] - f42_r[1] * r1 - f42_c[1] * c1;
    y_c[3] = f41_c[1] - f42_c[1] * r1 + f42_r[1] * c1;
    
    for(int i = 0; i<4; i++){
        y_r[i] = y_r[i]*0.5f;
        y_c[i] = y_c[i]*0.5f;
    }
}


// 16 point complex IFFT (__local variable version).
__device__ void ifft16_lvar(float4 *x_r, float4 *x_c, float4 *y_r, float4 *y_c)
{
    float4 t_r[2];
    float4 t_c[2];
    float4 f41_r[2];
    float4 f41_c[2];
    float4 f42_r[2];
    float4 f42_c[2];
     
    // 8 point IFFT.
    t_r[0] = make_float4(x_r[0].x, x_r[0].z, x_r[1].x, x_r[1].z);
    t_c[0] = make_float4(x_c[0].x, x_c[0].z, x_c[1].x, x_c[1].z);    
    t_r[1] = make_float4(x_r[2].x, x_r[2].z, x_r[3].x, x_r[3].z);
    t_c[1] = make_float4(x_c[2].x, x_c[2].z, x_c[3].x, x_c[3].z);    
    ifft8(t_r, t_c, f41_r, f41_c);

    t_r[0] = make_float4(x_r[0].y, x_r[0].w, x_r[1].y, x_r[1].w);
    t_c[0] = make_float4(x_c[0].y, x_c[0].w, x_c[1].y, x_c[1].w);
    t_r[1] = make_float4(x_r[2].y, x_r[2].w, x_r[3].y, x_r[3].w);
    t_c[1] = make_float4(x_c[2].y, x_c[2].w, x_c[3].y, x_c[3].w);
    ifft8(t_r, t_c, f42_r, f42_c);
        
    // Shift and add.
    float4 r1 = make_float4(1.0f,  9.23879533e-01f,  7.07106781e-01f,  3.82683432e-01f);
    float4 c1 = make_float4(0.0f, -3.82683432e-01f, -7.07106781e-01f, -9.23879533e-01f);
    y_r[0] = f41_r[0] + f42_r[0] * r1 + f42_c[0] * c1;
    y_c[0] = f41_c[0] + f42_c[0] * r1 - f42_r[0] * c1;    
    y_r[2] = f41_r[0] - f42_r[0] * r1 - f42_c[0] * c1;
    y_c[2] = f41_c[0] - f42_c[0] * r1 + f42_r[0] * c1;    
    
    r1 = make_float4( 0.0f, -3.82683432e-01f, -7.07106781e-01f, -9.23879533e-01f);
    c1 = make_float4(-1.0f, -9.23879533e-01f, -7.07106781e-01f, -3.82683432e-01f);
    y_r[1] = f41_r[1] + f42_r[1] * r1 + f42_c[1] * c1;
    y_c[1] = f41_c[1] + f42_c[1] * r1 - f42_r[1] * c1;    
    y_r[3] = f41_r[1] - f42_r[1] * r1 - f42_c[1] * c1;
    y_c[3] = f41_c[1] - f42_c[1] * r1 + f42_r[1] * c1;
    
    for(int i = 0; i<4; i++){
        y_r[i] = y_r[i]*0.5f;
        y_c[i] = y_c[i]*0.5f;    
    }
}


// 16 x 16 point complex FFT with work group size of 16.
__device__ void fft_16x16_wg16(float4 *x_r, float4 *x_c, float4 *y_r, float4 *y_c, int lid)
{
    int j;
    
    float4 t1_r[4];
    float4 t1_c[4];
    
    float *y1_r = (float *)y_r;
    float *y1_c = (float *)y_c;
    
    // Axis 1.
    fft16_lvar(&(x_r[lid*4]), &(x_c[lid*4]), &(y_r[lid*4]), &(y_c[lid*4]));
    
    __syncthreads();
    
    // Axis 2.
    
    // Convert columns to rows.
    for(j=0; j<4; j++){
        t1_r[j].x = y1_r[(4*j+0)*16+lid];
        t1_r[j].y = y1_r[(4*j+1)*16+lid];
        t1_r[j].z = y1_r[(4*j+2)*16+lid];
        t1_r[j].w = y1_r[(4*j+3)*16+lid];
        t1_c[j].x = y1_c[(4*j+0)*16+lid];
        t1_c[j].y = y1_c[(4*j+1)*16+lid];
        t1_c[j].z = y1_c[(4*j+2)*16+lid];
        t1_c[j].w = y1_c[(4*j+3)*16+lid];    
    }
        
    fft16(t1_r, t1_c, t1_r, t1_c);
        
    // Reverse conversion.
    for(j=0; j<4; j++){
        y1_r[(4*j+0)*16+lid] = t1_r[j].x;
        y1_r[(4*j+1)*16+lid] = t1_r[j].y;
        y1_r[(4*j+2)*16+lid] = t1_r[j].z;
        y1_r[(4*j+3)*16+lid] = t1_r[j].w;
        y1_c[(4*j+0)*16+lid] = t1_c[j].x;
        y1_c[(4*j+1)*16+lid] = t1_c[j].y;
        y1_c[(4*j+2)*16+lid] = t1_c[j].z;
        y1_c[(4*j+3)*16+lid] = t1_c[j].w;            
    }
    
    __syncthreads();
}


// 16 x 16 point complex IFFT with work group size of 16.
__device__ void ifft_16x16_wg16(float4 *x_r, float4 *x_c, float4 *y_r, float4 *y_c, int lid)
{
    int j;
    
    float4 t1_r[4];
    float4 t1_c[4];

    float *x1_r = (float *)x_r;
    float *x1_c = (float *)x_c;
    float *y1_r = (float *)y_r;
    float *y1_c = (float *)y_c;
    
    // Axis 2.
    
    // Convert columns to rows.
    for(j=0; j<4; j++){
        t1_r[j].x = x1_r[(4*j+0)*16+lid];
        t1_r[j].y = x1_r[(4*j+1)*16+lid];
        t1_r[j].z = x1_r[(4*j+2)*16+lid];
        t1_r[j].w = x1_r[(4*j+3)*16+lid];
        t1_c[j].x = x1_c[(4*j+0)*16+lid];
        t1_c[j].y = x1_c[(4*j+1)*16+lid];
        t1_c[j].z = x1_c[(4*j+2)*16+lid];
        t1_c[j].w = x1_c[(4*j+3)*16+lid];    
     }
        
     ifft16(t1_r, t1_c, t1_r, t1_c);
        
     // Reverse conversion.
     for(j=0; j<4; j++){
         y1_r[(4*j+0)*16+lid] = t1_r[j].x;
         y1_r[(4*j+1)*16+lid] = t1_r[j].y;
         y1_r[(4*j+2)*16+lid] = t1_r[j].z;
         y1_r[(4*j+3)*16+lid] = t1_r[j].w;
         y1_c[(4*j+0)*16+lid] = t1_c[j].x;
         y1_c[(4*j+1)*16+lid] = t1_c[j].y;
         y1_c[(4*j+2)*16+lid] = t1_c[j].z;
         y1_c[(4*j+3)*16+lid] = t1_c[j].w;        
    }
    
    __syncthreads();
    
    // Axis 1.
    ifft16_lvar(&(y_r[lid*4]), &(y_c[lid*4]), &(y_r[lid*4]), &(y_c[lid*4]));

    __syncthreads();
}


/******************
 * Vector functions.
 *
 * These are all designed on vectors with 64 float4 elements
 * and lid values in the range (0 - 15).
 *
 ******************/

__device__ void veccopy(float4 *v1, float4 *v2, int lid)
{
    int i = lid*4;

    v1[i]   = v2[i];
    v1[i+1] = v2[i+1];
    v1[i+2] = v2[i+2];
    v1[i+3] = v2[i+3];
}

__device__ void vecncopy(float4 *v1, float4 *v2, int lid)
{
    int i = lid*4;

    v1[i]   = -v2[i];
    v1[i+1] = -v2[i+1];
    v1[i+2] = -v2[i+2];
    v1[i+3] = -v2[i+3];
}

/* Returns the dot product as the first element of w1. */
__device__ void vecdot(float *w1, float4 *v1, float4 *v2, int lid)
{
    int i = lid*4;
    float sum = 0.0f;

    sum += dot(v1[i]  , v2[i]);
    sum += dot(v1[i+1], v2[i+1]);
    sum += dot(v1[i+2], v2[i+2]);
    sum += dot(v1[i+3], v2[i+3]);
    w1[lid] = sum;

    __syncthreads();

    if(lid == 0){
    	for(i=1; i<16; i++){
	   w1[0] += w1[i];
	}
    }

    __syncthreads();
}

/* Returns 0 or a positive integer as the first element of w1. */
__device__ void vecisEqual(int *w1, float4 *v1, float4 *v2, int lid)
{
    int i = lid*4;
    int sum = 0;

    sum += (v1[i].x   != v2[i].x)   + (v1[i].y   != v2[i].y)   + (v1[i].z   != v2[i].z)   + (v1[i].w   != v2[i].w);
    sum += (v1[i+1].x != v2[i+1].x) + (v1[i+1].y != v2[i+1].y) + (v1[i+1].z != v2[i+1].z) + (v1[i+1].w != v2[i+1].w);
    sum += (v1[i+2].x != v2[i+2].x) + (v1[i+2].y != v2[i+2].y) + (v1[i+2].z != v2[i+2].z) + (v1[i+2].w != v2[i+2].w);
    sum += (v1[i+3].x != v2[i+3].x) + (v1[i+3].y != v2[i+3].y) + (v1[i+3].z != v2[i+3].z) + (v1[i+3].w != v2[i+3].w);
    w1[lid] = sum;

    __syncthreads();

    if(lid == 0){
    	for(i=1; i<16; i++){
	   w1[0] += w1[i];
	}
	w1[0] = !w1[0];
    }

    __syncthreads();
}

/* v1 = v2 * s1 + v3 */
__device__ void vecfma(float4 *v1, float4 *v2, float4 *v3, float s1, int lid)
{
    int i = lid*4;

    v1[i].x   = fma(s1, v2[i].x, v3[i].x);
    v1[i].y   = fma(s1, v2[i].y, v3[i].y);
    v1[i].z   = fma(s1, v2[i].z, v3[i].z);
    v1[i].w   = fma(s1, v2[i].w, v3[i].w);
    i += 1;

    v1[i].x   = fma(s1, v2[i].x, v3[i].x);
    v1[i].y   = fma(s1, v2[i].y, v3[i].y);
    v1[i].z   = fma(s1, v2[i].z, v3[i].z);
    v1[i].w   = fma(s1, v2[i].w, v3[i].w);
    i += 1;

    v1[i].x   = fma(s1, v2[i].x, v3[i].x);
    v1[i].y   = fma(s1, v2[i].y, v3[i].y);
    v1[i].z   = fma(s1, v2[i].z, v3[i].z);
    v1[i].w   = fma(s1, v2[i].w, v3[i].w);
    i += 1;

    v1[i].x   = fma(s1, v2[i].x, v3[i].x);
    v1[i].y   = fma(s1, v2[i].y, v3[i].y);
    v1[i].z   = fma(s1, v2[i].z, v3[i].z);
    v1[i].w   = fma(s1, v2[i].w, v3[i].w);
}

/* v1 = v2 * s1 + v1 */
__device__ void vecfmaInplace(float4 *v1, float4 *v2, float s1, int lid)
{
    int i = lid*4;

    v1[i].x   = fma(s1, v2[i].x, v1[i].x);
    v1[i].y   = fma(s1, v2[i].y, v1[i].y);
    v1[i].z   = fma(s1, v2[i].z, v1[i].z);
    v1[i].w   = fma(s1, v2[i].w, v1[i].w);
    i += 1;

    v1[i].x   = fma(s1, v2[i].x, v1[i].x);
    v1[i].y   = fma(s1, v2[i].y, v1[i].y);
    v1[i].z   = fma(s1, v2[i].z, v1[i].z);
    v1[i].w   = fma(s1, v2[i].w, v1[i].w);
    i += 1;

    v1[i].x   = fma(s1, v2[i].x, v1[i].x);
    v1[i].y   = fma(s1, v2[i].y, v1[i].y);
    v1[i].z   = fma(s1, v2[i].z, v1[i].z);
    v1[i].w   = fma(s1, v2[i].w, v1[i].w);
    i += 1;

    v1[i].x   = fma(s1, v2[i].x, v1[i].x);
    v1[i].y   = fma(s1, v2[i].y, v1[i].y);
    v1[i].z   = fma(s1, v2[i].z, v1[i].z);
    v1[i].w   = fma(s1, v2[i].w, v1[i].w);
}

__device__ void vecnorm(float *w1, float4 *v1, int lid)
{
    vecdot(w1, v1, v1, lid);

    if(lid == 0){
        w1[0] = sqrt(w1[0]);
    }

    __syncthreads();
}

__device__ void vecscaleInplace(float4 *v1, float s1, int lid)
{
    int i = lid*4;
    
    float4 t1 = make_float4(s1, s1, s1, s1);
    v1[i]   = v1[i]*t1;
    v1[i+1] = v1[i+1]*t1;
    v1[i+2] = v1[i+2]*t1;
    v1[i+3] = v1[i+3]*t1;
}

__device__ void vecsub(float4 *v1, float4 *v2, float4 *v3, int lid)
{
    int i = lid*4;
    
    v1[i]   = v2[i]   - v3[i];
    v1[i+1] = v2[i+1] - v3[i+1];
    v1[i+2] = v2[i+2] - v3[i+2];
    v1[i+3] = v2[i+3] - v3[i+3];
}


/****************
 * NCS functions.
 *
 * These are all designed on vectors with 64 float4 elements
 * and lid values in the range (0 - 15).
 *
 ****************/

__device__ void calcLLGradient(float4 *u,
                               float4 *data,
		               float4 *gamma,
		               float4 *gradient,
		               int lid)
{
    int i = lid*4;
    float4 t1;
    float4 t2;

    t1 = data[i] + gamma[i];
    t2.x = fmax(u[i].x + gamma[i].x, FITMIN);
    t2.y = fmax(u[i].y + gamma[i].y, FITMIN);
    t2.z = fmax(u[i].z + gamma[i].z, FITMIN);
    t2.w = fmax(u[i].w + gamma[i].w, FITMIN);
    gradient[i] = make_float4(1.0f, 1.0f, 1.0f, 1.0f) - t1/t2;

    i += 1;
    t1 = data[i] + gamma[i];
    t2.x = fmax(u[i].x + gamma[i].x, FITMIN);
    t2.y = fmax(u[i].y + gamma[i].y, FITMIN);
    t2.z = fmax(u[i].z + gamma[i].z, FITMIN);
    t2.w = fmax(u[i].w + gamma[i].w, FITMIN);
    gradient[i] = make_float4(1.0f, 1.0f, 1.0f, 1.0f) - t1/t2;

    i += 1;
    t1 = data[i] + gamma[i];
    t2.x = fmax(u[i].x + gamma[i].x, FITMIN);
    t2.y = fmax(u[i].y + gamma[i].y, FITMIN);
    t2.z = fmax(u[i].z + gamma[i].z, FITMIN);
    t2.w = fmax(u[i].w + gamma[i].w, FITMIN);
    gradient[i] = make_float4(1.0f, 1.0f, 1.0f, 1.0f) - t1/t2;

    i += 1;
    t1 = data[i] + gamma[i];
    t2.x = fmax(u[i].x + gamma[i].x, FITMIN);
    t2.y = fmax(u[i].y + gamma[i].y, FITMIN);
    t2.z = fmax(u[i].z + gamma[i].z, FITMIN);
    t2.w = fmax(u[i].w + gamma[i].w, FITMIN);
    gradient[i] = make_float4(1.0f, 1.0f, 1.0f, 1.0f) - t1/t2;    
}

__device__ void calcLogLikelihood(float *w1,
                                  float4 *u,
		                  float4 *data,
		                  float4 *gamma,
		                  int lid)
{
    int i = lid*4;
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 t1;
    float4 t2;

    t1 = data[i] + gamma[i];
    t2.x = log(fmax(u[i].x + gamma[i].x, FITMIN));
    t2.y = log(fmax(u[i].y + gamma[i].y, FITMIN));
    t2.z = log(fmax(u[i].z + gamma[i].z, FITMIN));
    t2.w = log(fmax(u[i].w + gamma[i].w, FITMIN));
    sum += u[i] - t1*t2;

    i += 1;
    t1 = data[i] + gamma[i];
    t2.x = log(fmax(u[i].x + gamma[i].x, FITMIN));
    t2.y = log(fmax(u[i].y + gamma[i].y, FITMIN));
    t2.z = log(fmax(u[i].z + gamma[i].z, FITMIN));
    t2.w = log(fmax(u[i].w + gamma[i].w, FITMIN));
    sum += u[i] - t1*t2;

    i += 1;
    t1 = data[i] + gamma[i];
    t2.x = log(fmax(u[i].x + gamma[i].x, FITMIN));
    t2.y = log(fmax(u[i].y + gamma[i].y, FITMIN));
    t2.z = log(fmax(u[i].z + gamma[i].z, FITMIN));
    t2.w = log(fmax(u[i].w + gamma[i].w, FITMIN));
    sum += u[i] - t1*t2;

    i += 1;
    t1 = data[i] + gamma[i];
    t2.x = log(fmax(u[i].x + gamma[i].x, FITMIN));
    t2.y = log(fmax(u[i].y + gamma[i].y, FITMIN));
    t2.z = log(fmax(u[i].z + gamma[i].z, FITMIN));
    t2.w = log(fmax(u[i].w + gamma[i].w, FITMIN));
    sum += u[i] - t1*t2;

    w1[lid] = sum.x + sum.y + sum.z + sum.w;

    __syncthreads();
    
    if(lid == 0){
    	for(i=1; i<16; i++){
	   w1[0] += w1[i];
	}
    }

    __syncthreads();        
}

/* 
 * Use inverse FFT optimization.
 *
 * Notes:
 *
 *  1. Only gives correct values for valid OTFs, meaning
 *     that they are the FT of a realistic PSF.
 *
 *  2. The u_fft_r and u_fft_c parameters must also be the
 *     FT of a real valued image.
 */
__device__ void calcNCGradientIFFT(float4 *w1,
                                  float4 *w2,
		                  float4 *w3,
                                  float4 *u_fft_r, 
                                  float4 *u_fft_c, 
                                  float4 *otf_mask_sqr,
                                  float4 *gradient,
			          int lid)
{
    int i = lid*4;
    
    float4 t1;

    t1 = 2.0f*otf_mask_sqr[i];
    w1[i] = t1*u_fft_r[i];
    w2[i] = t1*u_fft_c[i];

    i += 1;
    t1 = 2.0f*otf_mask_sqr[i];
    w1[i] = t1*u_fft_r[i];
    w2[i] = t1*u_fft_c[i];

    i += 1;
    t1 = 2.0f*otf_mask_sqr[i];
    w1[i] = t1*u_fft_r[i];
    w2[i] = t1*u_fft_c[i];

    i += 1;
    t1 = 2.0f*otf_mask_sqr[i];
    w1[i] = t1*u_fft_r[i];
    w2[i] = t1*u_fft_c[i];

    __syncthreads();
    
    ifft_16x16_wg16(w1, w2, gradient, w3, lid);
}

__device__ void calcNoiseContribution(float *w1,
                                      float4 *u_fft_r,
			              float4 *u_fft_c,
			              float4 *otf_mask_sqr,
			              int lid)
{
    int i = lid*4;
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 t1;
    
    t1 = u_fft_r[i]*u_fft_r[i] + u_fft_c[i]*u_fft_c[i];
    sum += t1*otf_mask_sqr[i];

    i += 1;
    t1 = u_fft_r[i]*u_fft_r[i] + u_fft_c[i]*u_fft_c[i];
    sum += t1*otf_mask_sqr[i];

    i += 1;
    t1 = u_fft_r[i]*u_fft_r[i] + u_fft_c[i]*u_fft_c[i];
    sum += t1*otf_mask_sqr[i];

    i += 1;
    t1 = u_fft_r[i]*u_fft_r[i] + u_fft_c[i]*u_fft_c[i];
    sum += t1*otf_mask_sqr[i];

    w1[lid] = sum.x + sum.y + sum.z + sum.w;

    __syncthreads();
    
    if(lid == 0){
    	for(i=1; i<16; i++){
	   w1[0] += w1[i];
	}
	w1[0] = w1[0] * (1.0f/(4.0f*PSIZE));
    }

    __syncthreads();        
}


/******************
 * L-BFGS functions.
 ******************/

__device__ void converged(int *w1,
                          float *w2,
	                  float4 *x,
	                  float4 *g,
	                  int lid)
{
    float xnorm;
    float gnorm;
    
    vecnorm(w2, x, lid); 
    if (lid == 0){
        xnorm = fmax(w2[0], 1.0f);
    }

    __syncthreads();
    
    vecnorm(w2, g, lid);
    if (lid == 0){
        gnorm = w2[0];
	if ((gnorm/xnorm) > EPSILON){
	    w1[0] = 0;
        }
        else{
            w1[0] = 1;
        }
    }

    __syncthreads();
}


/****************
 * Kernels.
 ****************/

/*
 * Run NCS noise reduction on sub-regions.
 * 
 * Note: Any zero or negative values in the sub-regions should be
 *       set to a small positive value like 1.0.
 *
 * data_in - Sub-region data in e-.
 * g_gamma - Sub-region CMOS variance in units of e-^2.
 * otf_mask - 16 x 16 array containing the OTF mask.
 * data_out - Storage for noise corrected sub-regions.
 * iterations - Number of L-BFGS solver iterations.
 * status - Status of the solution (good, failed because of X).
 * alpha - NCS alpha term.
 */
extern "C" __global__ void ncsReduceNoise(float4 *data_in,
                                          float4 *g_gamma,
                                          float4 *otf_mask,
                                          float4 *data_out,
                                          int *g_iterations,
                                          int *g_status,
                                          float alpha)
{
    int gid = blockIdx.x;
    int lid = threadIdx.x;

    int i = lid*4;
    int i_g = gid*PSIZE;

    int j,k;
    
    __shared__ int bound;
    __shared__ int ci;
    __shared__ int searching;
  
    __shared__ float beta;
    __shared__ float cost;
    __shared__ float cost_p;
    __shared__ float step;
    __shared__ float t1;
    __shared__ float ys_c0;
    __shared__ float yy;

    __shared__ int w1_i[ASIZE];
    __shared__ float w1_f[ASIZE];
    
    __shared__ float a[M];
    __shared__ float ys[M];

    __shared__ float4 data[PSIZE];
    __shared__ float4 gamma[PSIZE];
    __shared__ float4 g_p[PSIZE];
    __shared__ float4 gradient[PSIZE];
    __shared__ float4 otf_mask_sqr[PSIZE];
    __shared__ float4 srch_dir[PSIZE];
    __shared__ float4 u_c[PSIZE];
    __shared__ float4 u_p[PSIZE];
    __shared__ float4 u_r[PSIZE]; 
    __shared__ float4 u_fft_r[PSIZE]; 
    __shared__ float4 u_fft_c[PSIZE];

    __shared__ float4 w1_f4[PSIZE];
    __shared__ float4 w2_f4[PSIZE];
    __shared__ float4 w3_f4[PSIZE];
    __shared__ float4 w4_f4[PSIZE];
    
    __shared__ float4 s[M][PSIZE];
    __shared__ float4 y[M][PSIZE];

    /* Initialization. */
    for (j=0; j<4; j++){
    	k = i+j;
        data[k] = data_in[i_g+k];
        gamma[k] = g_gamma[i_g+k];
        otf_mask_sqr[k] = otf_mask[k] * otf_mask[k];
        u_r[k] = data_in[i_g+k];
	u_c[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    __syncthreads();

    /* Calculate initial state. */
    
    /* Cost. */
    fft_16x16_wg16(u_r, u_c, u_fft_r, u_fft_c, lid);

    calcLogLikelihood(w1_f, u_r, data, gamma, lid);
    if (lid == 0){
        cost = w1_f[0];
    }

    __syncthreads();

    calcNoiseContribution(w1_f, u_fft_r, u_fft_c, otf_mask_sqr, lid);
    if (lid == 0){
        cost += alpha*w1_f[0];
    }

    __syncthreads();
    
    /* Gradient. */
    calcLLGradient(u_r, data, gamma, gradient, lid);
    calcNCGradientIFFT(w1_f4, w2_f4, w3_f4, u_fft_r, u_fft_c, otf_mask_sqr, w4_f4, lid);
    vecfmaInplace(gradient, w4_f4, alpha, lid);

    __syncthreads();
    
    /* Check if we've already converged. */
    converged(w1_i, w1_f, u_r, gradient, lid);
    if (w1_i[0]){
        if (lid == 0){
            g_iterations[gid] = 1;
            g_status[gid] = SUCCESS;
	}
	for (j=0; j<4; j++){
            k = i+j;
    	    data_out[i_g+k] = u_r[k];
        }
	return;
    }
    
    /* Initial search direction. */
    vecnorm(w1_f, gradient, lid);
    if (lid == 0){
        step = 1.0/w1_f[0];
    }
    vecncopy(srch_dir, gradient, lid);

    __syncthreads();
    
    /* Start search. */
    for (k=1; k<(MAXITERS+1); k++){
    
        /* 
         * Line search. 
         *
         * This checks the Armijo rule/condition.
         * https://en.wikipedia.org/wiki/Wolfe_conditions
         */
	vecdot(w1_f, srch_dir, gradient, lid);
	if (lid == 0){
            t1 = C_1 * w1_f[0];
        }

        __syncthreads();
	
        if (t1 > 0.0){
            /* Increasing gradient. Minimization failed. */
            if (lid == 0){
                g_iterations[gid] = k;
                g_status[gid] = INCREASING_GRADIENT;
	    }
	    for (j=0; j<4; j++){
                k = i+j;
    	        data_out[i_g+k] = u_r[k];
            }
	    return;
        }
        
        /* Store current cost, u and gradient. */
	if (lid == 0){
            cost_p = cost;
	}
        veccopy(u_p, u_r, lid);
        veccopy(g_p, gradient, lid);
	
	/* Search for a good step size. */
	if (lid == 0){
            searching = 1;
	}

        __syncthreads();
	
        while(searching){
        
            /* Move in search direction. */
            vecfma(u_r, srch_dir, u_p, step, lid);

            __syncthreads();

            fft_16x16_wg16(u_r, u_c, u_fft_r, u_fft_c, lid);

            /* Calculate new cost. */
            calcLogLikelihood(w1_f, u_r, data, gamma, lid);
            if (lid == 0){
                cost = w1_f[0];
            }

            __syncthreads();

            calcNoiseContribution(w1_f, u_fft_r, u_fft_c, otf_mask_sqr, lid);
            if (lid == 0){
                cost += alpha*w1_f[0];
            }

            __syncthreads();
            
            /* Armijo condition. */
            if (cost <= (cost_p + t1*step)){
	        if (lid == 0){
                    searching = 0;
		}

                __syncthreads();
		
            }
            else{
	        if (lid == 0){ 
                    step = STEPM*step;
		}

                __syncthreads();
		
                if (step < MIN_STEP){
                    /* 
                     * Reached minimum step size. Minimization failed. 
                     * Return the last good u values.
                     */
                    if (lid == 0){
                        g_iterations[gid] = k+1;
                        g_status[gid] = MINIMUM_STEP;
	            }
		    for (j=0; j<4; j++){
                         k = i+j;
    	                 data_out[i_g+k] = u_p[k];
                    }
                    return;
                }
            }
        }
        
        /* Calculate new gradient. */
	calcLLGradient(u_r, data, gamma, gradient, lid);
        calcNCGradientIFFT(w1_f4, w2_f4, w3_f4, u_fft_r, u_fft_c, otf_mask_sqr, w4_f4, lid);
        vecfmaInplace(gradient, w4_f4, alpha, lid);

        __syncthreads();
    
        /* Convergence check. */
        converged(w1_i, w1_f, u_r, gradient, lid);
        if (w1_i[0]){
            if (lid == 0){
                g_iterations[gid] = k+1;
                g_status[gid] = SUCCESS;
	    }
            for (j=0; j<4; j++){
                k = i+j;
    	        data_out[i_g+k] = u_r[k];
            }
	    return;
        }
        
        /*
	 * Machine precision check.
	 *
	 * This is probably not an actual failure, we just ran out of digits. Reaching
	 * this state has a cost so we want to know if this is happening a lot.
	 */
	vecisEqual(w1_i, u_r, u_p, lid);
        if (w1_i[0]){
	    if (lid == 0){
                g_iterations[gid] = k+1;
                g_status[gid] = REACHED_MAXPRECISION;
            }
            for (j=0; j<4; j++){
                k = i+j;
    	        data_out[i_g+k] = u_r[k];
            }	    
            return;
        }
        
        /* L-BFGS calculation of new search direction. */
	if (lid == 0){
            ci = (k-1)%M;
	}

        __syncthreads();
	
        vecsub(s[ci], u_r, u_p, lid);
        vecsub(y[ci], gradient, g_p, lid);

        __syncthreads();

        vecdot(w1_f, s[ci], y[ci], lid);
	if (lid == 0){
            ys_c0 = w1_f[0];
            ys[ci] = 1.0/ys_c0;
	}

        __syncthreads(); 

        vecdot(w1_f, y[ci], y[ci], lid);
        if (lid == 0){
	    yy = 1.0/w1_f[0];
        }

        vecncopy(srch_dir, gradient, lid);

        if (lid == 0){
            bound = min(k, M);
	}

        __syncthreads();
	
        for(j=0; j<bound; j++){

            if (lid == 0){
	        ci = (k - j - 1)%M;
	    }
	    
            __syncthreads();

            vecdot(w1_f, s[ci], srch_dir, lid);
            if (lid == 0){
                a[ci] = w1_f[0]*ys[ci];
            }

            __syncthreads();

            vecfmaInplace(srch_dir, y[ci], -a[ci], lid);
        }
        
        vecscaleInplace(srch_dir, ys_c0*yy, lid);
        
        for(j=0; j<bound; j++){

            if (lid == 0){
	        ci = (k + j - bound)%M;
	    }

            __syncthreads();

            vecdot(w1_f, y[ci], srch_dir, lid);
	    if (lid == 0){
	        beta = w1_f[0]*ys[ci];
	    }

            __syncthreads();

            vecfmaInplace(srch_dir, s[ci], (a[ci] - beta), lid);
        }

        if (lid == 0){
            step = 1.0;
	}

        __syncthreads();
    }
    
    /* Reached maximum iterations. Minimization failed. */
    if (lid == 0){
        g_iterations[gid] = MAXITERS;
        g_status[gid] = REACHED_MAXITERS;
    }
    for (j=0; j<4; j++){
        k = i+j;
    	data_out[i_g+k] = u_r[k];
    }
}

