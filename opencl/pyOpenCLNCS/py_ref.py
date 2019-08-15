#!/usr/bin/env python
#
# Pure Python reference version for testing.
#
# Written to match the OpenCL code, not for style..
#
# Unfortunately the match is not perfect. I tried to use numpy.float32
# for all the math but there are still small differences..
#
# Hazen 08/19
#

import numpy

## Constants
FITMIN = numpy.float32(1.0e-6)
PSIZE = 64

## L-BFGS solver parameters.
C_1  = numpy.float32(1.0e-4)
EPSILON = numpy.float32(1.0e-4)
MAXITERS = 200
M = 8
MIN_STEP = numpy.float32(1.0e-6)
STEPM = numpy.float32(0.5)

## Error status codes.
SUCCESS = 0
UNSTARTED = -1
REACHED_MAXITERS = -2
INCREASING_GRADIENT = -3
MINIMUM_STEP = -4
REACHED_MAXPRECISION = -5


## FFT functions.

def fft_16(x_r, x_c, y_r, y_c):
    x_fft = numpy.fft.fft(x_r + 1j*x_c);
    y_r[:] = numpy.real(x_fft)
    y_c[:] = numpy.imag(x_fft)
    
def fft_16x16(x_r, x_c, y_r, y_c):
    t1 = x_r.reshape(16,16)
    t2 = x_c.reshape(16,16)
    x_fft = numpy.fft.fft2(t1 + 1j*t2)
    y_r[:] = numpy.real(x_fft).flatten()
    y_c[:] = numpy.imag(x_fft).flatten()

    
## Vector functions.

def veccopy(v1, v2):
    v1[:] = v2

def vecncopy(v1, v2):
    v1[:] = -v2

def vecdot(v1, v2):
    return numpy.sum(v1 * v2)

def vecisEqual(v1, v2):
    return numpy.array_equal(v1, v2)

def vecfma(v1, v2, v3, s1):
    v1[:] = v2*s1 + v3

def vecfmaInplace(v1, v2, s1):
    v1[:] = v2*s1 + v1

def vecmul(v1, v2, v3):
    v1[:] = v2*v3

def vecnorm(v1):
    return numpy.sqrt(vecdot(v1, v1))

def vecscaleInplace(v1, s1):
    v1[:] = v1*s1

def vecsub(v1, v2, v3):
    v1[:] = v2 - v3


## NCS functions.

def calcLLGradient(u, data, gamma, gradient):
    t1 = data + gamma
    t2 = u + gamma
    t2[(t2 < FITMIN)] = FITMIN
    gradient[:] = 1.0 - t1/t2

def calcLogLikelihood(u, data, gamma):
    t1 = data + gamma
    t2 = u + gamma
    t2[(t2 < FITMIN)] = FITMIN
    t2 = numpy.log(t2)
    
    return numpy.sum(u - t1*t2)

# This version matches the approach used in NCS/clib
def calcNCGradient(u_fft_grad_r, u_fft_grad_c, u_fft_r, u_fft_c, otf_mask_sqr, gradient):
    for i in range(PSIZE*4):
        t1 = u_fft_r * u_fft_grad_r[i] + u_fft_c * u_fft_grad_c[i]
        t1 = 2.0 * t1 * otf_mask_sqr
        gradient[i] = numpy.sum(t1)/(4.0 * PSIZE)

def calcNCGradientIFFT(u_fft_r, u_fft_c, otf_mask_sqr, gradient):
    u = (u_fft_r + 1j*u_fft_c).reshape(16, 16)
    u = 2.0 * u * otf_mask_sqr.reshape(16, 16)
    x = numpy.fft.ifft2(u)
    gradient[:] = numpy.real(x.flatten())

def calcNoiseContribution(u_fft_r, u_fft_c, otf_mask_sqr):
    t1 = u_fft_r * u_fft_r + u_fft_c * u_fft_c
    t1 = t1 * otf_mask_sqr
    return numpy.sum(t1)/(4.0 * PSIZE)

def converged(x, g):
    xnorm = max(vecnorm(x), 1.0)
    gnorm = vecnorm(g)
    if ((gnorm/xnorm) > EPSILON):
        return False
    else:
        return True


## Python specific helper functions.

def createOTFMask():
    x = (numpy.arange(16)-8.0)/3.0
    gx = numpy.exp(-x*x)
    rc_filter = 1.0 - numpy.outer(gx, gx)
    rc_filter = rc_filter - numpy.min(rc_filter)
    rc_filter = rc_filter/numpy.max(rc_filter)
    rc_filter = numpy.fft.fftshift(rc_filter).flatten()

    return rc_filter.astype(numpy.float32)
    
def createUFFTGrad():
    u_fft_grad_r = []
    u_fft_grad_c = []

    for i in range(4*PSIZE):
        u_fft_grad_r.append(None)
        u_fft_grad_c.append(None)

    initUFFTGrad(u_fft_grad_r, u_fft_grad_c)
    return [u_fft_grad_r, u_fft_grad_c]


## "Kernels"

def initUFFTGrad(u_fft_grad_r, u_fft_grad_c):
    x_r = numpy.zeros(4*PSIZE, dtype = numpy.float32)
    x_c = numpy.zeros(4*PSIZE, dtype = numpy.float32)
    y_r = numpy.zeros(4*PSIZE, dtype = numpy.float32)
    y_c = numpy.zeros(4*PSIZE, dtype = numpy.float32)
                
    for i in range(4*PSIZE):
        x_r[i] = 1.0
        fft_16x16(x_r, x_c, y_r, y_c)
        u_fft_grad_r[i] = numpy.copy(y_r)
        u_fft_grad_c[i] = numpy.copy(y_c)
        x_r[i] = 0.0
        
def ncsReduceNoise(u_fft_grad_r,
                   u_fft_grad_c,
                   data_in,
                   g_gamma,
                   otf_mask,
                   data_out,
                   iterations,
                   status,
                   alpha):

    g_id = 0
    status[g_id] = UNSTARTED
    offset = g_id*PSIZE

    a = numpy.zeros(M, dtype = numpy.float32)
    ys = numpy.zeros(M, dtype = numpy.float32)
    
    data = numpy.zeros(4*PSIZE, dtype = numpy.float32)
    gamma = numpy.zeros(4*PSIZE, dtype = numpy.float32)
    g_p = numpy.zeros(4*PSIZE, dtype = numpy.float32)
    gradient = numpy.zeros(4*PSIZE, dtype = numpy.float32)
    otf_mask_sqr  = numpy.zeros(4*PSIZE, dtype = numpy.float32)
    srch_dir = numpy.zeros(4*PSIZE, dtype = numpy.float32)
    u_r = numpy.zeros(4*PSIZE, dtype = numpy.float32)
    u_c = numpy.zeros(4*PSIZE, dtype = numpy.float32)
    u_fft_r = numpy.zeros(4*PSIZE, dtype = numpy.float32)
    u_fft_c = numpy.zeros(4*PSIZE, dtype = numpy.float32)
    u_p = numpy.zeros(4*PSIZE, dtype = numpy.float32)
    work1  = numpy.zeros(4*PSIZE, dtype = numpy.float32)

    s = []
    y = []
    for i in range(M):
        s.append(numpy.zeros(4*PSIZE, dtype = numpy.float32))
        y.append(numpy.zeros(4*PSIZE, dtype = numpy.float32))

    # Initialization.
    for i in range(PSIZE*4):
        data[i] = data_in.flatten()[i + offset]
        gamma[i] = g_gamma.flatten()[i]
        otf_mask_sqr[i] = otf_mask.flatten()[i] * otf_mask.flatten()[i]
        u_r[i] = data_in.flatten()[i + offset]
        u_c[i] = 0.0
    
    # Calculate initial state.
    fft_16x16(u_r, u_c, u_fft_r, u_fft_c)
    
    # Cost.
    cost = calcLogLikelihood(u_r, data, gamma)
    cost += alpha * calcNoiseContribution(u_fft_r, u_fft_c, otf_mask_sqr)
    
    # Gradient.
    calcLLGradient(u_r, data, gamma, gradient);
    calcNCGradient(u_fft_grad_r, u_fft_grad_c, u_fft_r, u_fft_c, otf_mask_sqr, work1);
    vecfmaInplace(gradient, work1, alpha);
    
    # Check if we've already converged.
    if (converged(u_r, gradient)):
        for i in range(PSIZE*4):
            data_out[i + offset] = u_r[i]
        iterations[g_id] = 1
        status[g_id] = SUCCESS
        return
    
    # Initial search direction.
    step = 1.0/vecnorm(gradient)
    vecncopy(srch_dir, gradient)

    # Start search.
    for k in range(MAXITERS):
    
        # 
        # Line search. 
        #
        # This checks the Armijo rule/condition.
        # https://en.wikipedia.org/wiki/Wolfe_conditions
        #
        t1 = C_1 * vecdot(srch_dir, gradient)
         
        if (t1 > 0.0):
            # Increasing gradient. Minimization failed.
            for i in range(PSIZE*4):
                data_out[i + offset] = u_r[i]
            iterations[g_id] = k+1
            status[g_id] = INCREASING_GRADIENT
            return
        
        # Store current cost, u and gradient.
        cost_p = cost
        veccopy(u_p, u_r)
        veccopy(g_p, gradient)

	# Search for a good step size.
        searching = 1
        while(searching):
        
            # Move in search direction.
            vecfma(u_r, srch_dir, u_p, step)
            
            # Calculate new cost.
            fft_16x16(u_r, u_c, u_fft_r, u_fft_c)
            cost = calcLogLikelihood(u_r, data, gamma)
            cost += alpha * calcNoiseContribution(u_fft_r, u_fft_c, otf_mask_sqr)
            
            # Armijo condition.
            if (cost <= (cost_p + t1*step)):
                searching = 0;
            else:
                step = STEPM*step
                if (step < MIN_STEP):
                    #
                    # Reached minimum step size. Minimization failed. 
                    # Return the last good u values.
                    #
                    for i in range(PSIZE*4):
                        data_out[i + offset] = u_p[i]
                    iterations[g_id] = k+1
                    status[g_id] = MINIMUM_STEP
                    return
        
        # Calculate new gradient.
        calcLLGradient(u_r, data, gamma, gradient)
        calcNCGradient(u_fft_grad_r, u_fft_grad_c, u_fft_r, u_fft_c, otf_mask_sqr, work1)
        vecfmaInplace(gradient, work1, alpha)
        
        # Convergence check.
        if (converged(u_r, gradient)):
            for i in range(PSIZE*4):
                data_out[i + offset] = u_r[i]
            iterations[g_id] = k+1
            status[g_id] = SUCCESS
            return
        
        #
	# Machine precision check.
        #
	# This is probably not an actual failure, we just ran out of digits. Reaching
	# this state has a cost so we want to know if this is happening a lot.
        #
        if vecisEqual(u_r, u_p):
            for i in range(PSIZE*4):
                data_out[i + offset] = u_r[i]
            iterations[g_id] = k+1
            status[g_id] = REACHED_MAXPRECISION
            return
        
        # L-BFGS calculation of new search direction.
        ci = (k-1)%M
        vecsub(s[ci], u_r, u_p)
        vecsub(y[ci], gradient, g_p)
        
        ys_c0 = vecdot(s[ci], y[ci])
        ys[ci] = 1.0/ys_c0
        yy = 1.0/vecdot(y[ci], y[ci])
        
        vecncopy(srch_dir, gradient)
        bound = min(k, M)
        for j in range(bound):
            ci = (k - j - 1)%M
            a[ci] = vecdot(s[ci], srch_dir)*ys[ci]
            vecfmaInplace(srch_dir, y[ci], -a[ci])
        
        vecscaleInplace(srch_dir, ys_c0*yy)
        
        for j in range(bound):
            ci = (k + j - bound)%M
            beta = vecdot(y[ci], srch_dir)*ys[ci]
            vecfmaInplace(srch_dir, s[ci], (a[ci] - beta))
        
        step = 1.0
    
    # Reached maximum iterations. Minimization failed.
    for i in range(PSIZE*4):
        data_out[i + offset] = u_r[i]
    iterations[g_id] = MAXITERS
    status[g_id] = REACHED_MAXITERS
    
