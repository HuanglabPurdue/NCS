#!/usr/bin/env python
#
# Test noise contribution calculations.
#
# This requires the pyCNCS module.
#
# Hazen 07/19
#
import numpy
import pyopencl as cl

# python3 and C NCS reference version.
import pyCNCS.ncs_c as ncsC

import pyOpenCLNCS
import pyOpenCLNCS.py_ref as pyRef

kernel_code = """

__kernel void calc_nc_test(__global float4 *g_u,
                           __global float4 *g_otf_mask, 
                           __global float *g_sum)
{
    int lid = get_local_id(0);
    int i = lid*4;
 
    __local float w1[16];
    __local float4 u_fft_r[PSIZE];
    __local float4 u_fft_c[PSIZE];
    __local float4 u_r[PSIZE];
    __local float4 u_c[PSIZE];
    __local float4 otf_mask_sqr[PSIZE];
    
    for(int j=0; j<4; j++){
        u_r[i+j] = g_u[i+j];
        u_c[i+j] = (float4)(0.0, 0.0, 0.0, 0.0);
        otf_mask_sqr[i+j] = g_otf_mask[i+j]*g_otf_mask[i+j];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    fft_16x16_wg16(u_r, u_c, u_fft_r, u_fft_c, lid);

    calcNoiseContribution(w1, u_fft_r, u_fft_c, otf_mask_sqr, lid);
    *g_sum = w1[0];
}

__kernel void calc_nc_grad_test(__global float4 *g_u,
                                __global float4 *g_otf_mask, 
                                __global float4 *g_gradient)
{
    int lid = get_local_id(0);
    int i = lid*4;

    __local float4 w1[PSIZE];
    __local float4 w2[PSIZE];
    __local float4 w3[PSIZE];
    __local float4 u_fft_r[PSIZE];
    __local float4 u_fft_c[PSIZE];
    __local float4 u_r[PSIZE];
    __local float4 u_c[PSIZE];
    __local float4 otf_mask_sqr[PSIZE];
    __local float4 gradient[PSIZE];

    for(int j=0; j<4; j++){
        u_r[i+j] = g_u[i+j];
        u_c[i+j] = (float4)(0.0, 0.0, 0.0, 0.0);
        otf_mask_sqr[i+j] = g_otf_mask[i+j]*g_otf_mask[i+j];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    fft_16x16_wg16(u_r, u_c, u_fft_r, u_fft_c, lid);

    calcNCGradientIFFT(w1, w2, w3, u_fft_r, u_fft_c, otf_mask_sqr, gradient, lid);

    for(int j=0; j<4; j++){
        g_gradient[i+j] = gradient[i+j];
    }
}

"""

#
# OpenCL setup.
#
kernel_code = pyOpenCLNCS.loadNCSKernel() + kernel_code

# Create context and command queue
platform = cl.get_platforms()[0]
devices = platform.get_devices()
context = cl.Context(devices)
queue = cl.CommandQueue(context,
                        properties=cl.command_queue_properties.PROFILING_ENABLE)

# Open program file and build
program = cl.Program(context, kernel_code)
try:
   program.build()
except:
   print("Build log:")
   print(program.get_build_info(devices[0], 
         cl.program_build_info.LOG))
   raise

def test_calc_nc():
   n_pts = 16

   for i in range(100):
    
      # OpenCL
      u = numpy.random.uniform(low = 1.0, high = 10.0, size = (n_pts, n_pts)).astype(dtype = numpy.float32)
      otf_mask_shift = pyRef.createOTFMask()

      nc = numpy.zeros(1, dtype = numpy.float32)

      u_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = u)
      otf_mask_buffer = cl.Buffer(context, 
                                  cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                                  hostbuf = otf_mask_shift)
      nc_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = nc)
      
      program.calc_nc_test(queue, (16,), (16,), u_buffer, otf_mask_buffer, nc_buffer)
      cl.enqueue_copy(queue, nc, nc_buffer).wait()
      queue.finish()

      # Reference 1
      otf_mask = numpy.fft.fftshift(otf_mask_shift.reshape(16, 16))
      ncs_sr = ncsC.NCSCSubRegion(r_size = n_pts)
      ncs_sr.setOTFMask(otf_mask)
      ncs_sr.setU(u)
      ref1_nc = ncs_sr.calcNoiseContribution()
      ncs_sr.cleanup()

      norm_diff = abs(nc[0] - ref1_nc)/abs(ref1_nc)    
      assert (norm_diff < 1.0e-3), "Difference in results! {0:.6f}".format(norm_diff)

      # Reference 2
      u_r = numpy.copy(u).flatten()
      u_c = numpy.zeros_like(u_r)
      u_fft_r = numpy.zeros_like(u_r)
      u_fft_c = numpy.zeros_like(u_c)
      otf_mask_sqr = (otf_mask_shift * otf_mask_shift).flatten()
      
      pyRef.fft_16x16(u_r, u_c, u_fft_r, u_fft_c)
      ref2_nc = pyRef.calcNoiseContribution(u_fft_r, u_fft_c, otf_mask_sqr)
      
      norm_diff = abs(nc[0] - ref2_nc)/abs(ref2_nc)
      assert (norm_diff < 1.0e-3), "Difference in results! {0:.6f}".format(norm_diff)

def test_calc_nc_grad_1():
   n_pts = 16
   
   for i in range(10):
      
      # OpenCL gradient calculation.
      u = numpy.random.uniform(low = 1.0, high = 10.0, size = (n_pts, n_pts)).astype(dtype = numpy.float32)
      otf_mask_shift = pyRef.createOTFMask()
      grad = numpy.zeros((n_pts, n_pts)).astype(numpy.float32)
      
      u_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = u)
      otf_mask_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = otf_mask_shift)
      grad_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = grad)
      
      program.calc_nc_grad_test(queue, (16,), (16,),
                                u_buffer,
                                otf_mask_buffer,
                                grad_buffer) 

      cl.enqueue_copy(queue, grad, grad_buffer).wait()
      queue.finish()
      
      # Reference 1
      otf_mask = numpy.fft.fftshift(otf_mask_shift.reshape(16, 16))
      ncs_sr = ncsC.NCSCSubRegion(r_size = n_pts)
      ncs_sr.setOTFMask(otf_mask)
      ncs_sr.setU(u)
      ncs_sr.calcNoiseContribution()
      ref1_grad = ncs_sr.calcNCGradient().reshape(grad.shape)
      ncs_sr.cleanup()
      
      ref_norm = numpy.abs(ref1_grad)
      ref_norm[(ref_norm<1.0)] = 1.0

      max_diff = numpy.max(numpy.abs(grad - ref1_grad)/ref_norm)
      assert (max_diff < 1.0e-5), "Difference in results! {0:.8f}".format(max_diff)

      # Reference 2
      u_r = numpy.copy(u).flatten()
      u_c = numpy.zeros_like(u_r)
      u_fft_r = numpy.zeros_like(u_r)
      u_fft_c = numpy.zeros_like(u_r)
      ref2_grad = numpy.zeros_like(u_r)
      otf_mask_sqr = otf_mask_shift * otf_mask_shift
      
      pyRef.fft_16x16(u_r, u_c, u_fft_r, u_fft_c)
      pyRef.calcNCGradientIFFT(u_fft_r, u_fft_c, otf_mask_sqr, ref2_grad)

      ref_norm = numpy.abs(ref2_grad)
      ref_norm[(ref_norm<1.0)] = 1.0

      max_diff = numpy.max(numpy.abs(grad.flatten() - ref2_grad)/ref_norm)
      assert (max_diff < 1.0e-5), "Difference in results! {0:.8f}".format(max_diff)
      

if (__name__ == "__main__"):
   test_calc_nc_grad_1()
   
