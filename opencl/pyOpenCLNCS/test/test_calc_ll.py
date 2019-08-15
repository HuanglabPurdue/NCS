#!/usr/bin/env python
#
# Test log-likelihood calculations.
#
# This requires the pyCNCS module.
#
# Hazen 07/19
#
import numpy
import pyopencl as cl

# python3 and C NCS reference version.
import pyCNCS.ncs_c as ncsC
import pyOpenCLNCS.py_ref as pyRef

import pyOpenCLNCS

kernel_code = """

__kernel void cll_test(__global float4 *g_u,
                       __global float4 *g_data, 
                       __global float4 *g_gamma,
                       __global float *g_sum) 
{
    float4 u[PSIZE];
    float4 data[PSIZE];
    float4 gamma[PSIZE];
    
    for(int i=0; i<PSIZE; i++){
        u[i] = g_u[i];
        data[i] = g_data[i];
        gamma[i] = g_gamma[i];
    }
    
    *g_sum = calcLogLikelihood(u, data, gamma);
}

__kernel void cll_grad_test(__global float4 *g_u,
                            __global float4 *g_data, 
                            __global float4 *g_gamma,
                            __global float4 *g_grad) 
{
    float4 u[PSIZE];
    float4 data[PSIZE];
    float4 gamma[PSIZE];
    float4 grad[PSIZE];
    
    for(int i=0; i<PSIZE; i++){
        u[i] = g_u[i];
        data[i] = g_data[i];
        gamma[i] = g_gamma[i];
    }
    
    calcLLGradient(u, data, gamma, grad);

    for(int i=0; i<PSIZE; i++){
        g_grad[i] = grad[i];
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

def test_calc_ll():
   n_pts = 16

   for i in range(100):
   
      # OpenCL
      u = numpy.random.uniform(low = -2.0, high = 10.0, size = (n_pts, n_pts)).astype(dtype = numpy.float32)
      data = numpy.random.uniform(low = 1.0, high = 10.0, size = (n_pts, n_pts)).astype(dtype = numpy.float32)
      gamma = numpy.random.uniform(low = 1.0, high = 2.0, size = (n_pts, n_pts)).astype(dtype = numpy.float32)
      ll = numpy.zeros(1, dtype = numpy.float32)
   
      u_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = u)
      data_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = data)
      gamma_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = gamma)
      ll_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = ll)

      program.cll_test(queue, (1,), (1,), u_buffer, data_buffer, gamma_buffer, ll_buffer)
      cl.enqueue_copy(queue, ll, ll_buffer).wait()
      queue.finish()
      
      # Reference 1
      ncs_sr = ncsC.NCSCSubRegion(r_size = n_pts)
      ncs_sr.newRegion(data, gamma)
      ncs_sr.setU(u)
      ref1_ll = ncs_sr.calcLogLikelihood()
      ncs_sr.cleanup()
      
      assert (abs(ll[0] - ref1_ll) < 1.0e-3), "Difference in results! {0:.6f} {1:.6f}".format(ll[0], ref1_ll)

      # Reference 2
      ref2_ll = pyRef.calcLogLikelihood(u, data, gamma)
      assert (abs(ll[0] - ref2_ll) < 1.0e-3), "Difference in results! {0:.6f} {1:.6f}".format(ll[0], ref2_ll)


def test_calc_ll_grad():
   n_pts = 16

   for i in range(100):

      # OpenCL
      u = numpy.random.uniform(low = -2.0, high = 10.0, size = (n_pts, n_pts)).astype(dtype = numpy.float32)
      data = numpy.random.uniform(low = 1.0, high = 10.0, size = (n_pts, n_pts)).astype(dtype = numpy.float32)
      gamma = numpy.random.uniform(low = 1.0, high = 2.0, size = (n_pts, n_pts)).astype(dtype = numpy.float32)
      grad = numpy.zeros((n_pts, n_pts), dtype = numpy.float32)
    
      u_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = u)
      data_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = data)
      gamma_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = gamma)
      grad_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = grad)
      
      program.cll_grad_test(queue, (1,), (1,), u_buffer, data_buffer, gamma_buffer, grad_buffer)
      cl.enqueue_copy(queue, grad, grad_buffer).wait()
      queue.finish()
      
      # Reference
      ncs_sr = ncsC.NCSCSubRegion(r_size = n_pts)
      ncs_sr.newRegion(data, gamma)
      ncs_sr.setU(u)
      ref1_grad = ncs_sr.calcLLGradient().reshape(grad.shape)
      ncs_sr.cleanup()
 
      ref1_norm = numpy.abs(ref1_grad)
      ref1_norm[(ref1_norm<1.0)] = 1.0

      max_diff = numpy.max(numpy.abs(grad - ref1_grad)/ref1_norm)
      assert (max_diff < 1.0e-5), "Difference in results! {0:.8f}".format(max_diff)

      ref2_grad = numpy.zeros((n_pts, n_pts), dtype = numpy.float32)
      pyRef.calcLLGradient(u, data, gamma, ref2_grad)

      ref2_norm = numpy.abs(ref2_grad)
      ref2_norm[(ref2_norm<1.0)] = 1.0

      max_diff = numpy.max(numpy.abs(grad - ref2_grad)/ref2_norm)
      assert (max_diff < 1.0e-5), "Difference in results! {0:.8f}".format(max_diff)
      
