#!/usr/bin/env python
#
# Test solver functions.
#
# Hazen 07/19
#
import numpy
import pyopencl as cl

import pyOpenCLNCS

kernel_code = """

__kernel void converged_test(__global float4 *g_v1,
                             __global float4 *g_v2,
                             __global int *g_conv)
{
    float4 v1[PSIZE];
    float4 v2[PSIZE];
    
    for(int i=0; i<PSIZE; i++){
        v1[i] = g_v1[i];
        v2[i] = g_v2[i];
    }
    
    *g_conv = converged(v1, v2);
}

__kernel void moduloM_test(__global int *g_i,
                           __global int *g_j)
{
    int i = g_i[0];
    *g_j = moduloM(i);
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

n_pts = 256

def test_converged_1():
   
   # Test 1 (not converged).
   v1 = 0.9*numpy.ones(n_pts).astype(numpy.float32)
   v2 = 1.0e-4*numpy.ones(n_pts).astype(numpy.float32)
   v3 = numpy.array([-1]).astype(numpy.int32)

   v1_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
   v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)
   v3_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v3)
   
   program.converged_test(queue, (1,), (1,), v1_buffer, v2_buffer, v3_buffer)
   cl.enqueue_copy(queue, v3, v3_buffer).wait()
   queue.finish()
   
   assert (v3[0] == 0)

def test_converged_2():
   # Test 2 (converged).
   v1 = 1.1*numpy.ones(n_pts).astype(numpy.float32)
   v2 = 1.0e-4*numpy.ones(n_pts).astype(numpy.float32)
   v3 = numpy.array([-1]).astype(numpy.int32)
   
   v1_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
   v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)
   v3_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v3)
   
   program.converged_test(queue, (1,), (1,), v1_buffer, v2_buffer, v3_buffer)
   cl.enqueue_copy(queue, v3, v3_buffer).wait()
   queue.finish()

   assert (v3[0] == 1)

def test_converged_3():
   # Test 3 (converged due xnorm minimum of 1.0).
   v1 = (0.1/numpy.sqrt(n_pts))*numpy.ones(n_pts).astype(numpy.float32)
   v2 = (0.9e-4/numpy.sqrt(n_pts))*numpy.ones(n_pts).astype(numpy.float32)
   v3 = numpy.array([-1]).astype(numpy.int32)
   
   v1_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
   v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)
   v3_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v3)
   
   program.converged_test(queue, (1,), (1,), v1_buffer, v2_buffer, v3_buffer)
   cl.enqueue_copy(queue, v3, v3_buffer).wait()
   queue.finish()
   
   assert (v3[0] == 1)

def _test_moduloM():
   #
   # Not used. I'd read that module on a GPU was very slow and should be
   # avoided so I tried replacing it with a bitwise AND. This however
   # had no effect on the processing time so I went back to modulo as it
   # is more flexible.
   #
   
   # Figure out value of M in kernel_code
   for elt in kernel_code.splitlines():
      if elt.startswith('#define M '):
          m_val = int(elt.split(" ")[2])
          assert ((m_val > 0) and ((m_val & (m_val - 1)) == 0)), str(m_val) + " is not a power of 2!"
   
   for i in range(20):
      v1 = numpy.array([i]).astype(numpy.int)
      v2 = numpy.zeros(1, dtype = numpy.int)

      v1_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
      v2_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)
   
      program.moduloM_test(queue, (1,), (1,), v1_buffer, v2_buffer)
      cl.enqueue_copy(queue, v2, v2_buffer).wait()
      queue.finish()
   
      assert (v2[0] == (i % 8))
      
if (__name__ == "__main__"):
   test_moduloM()
   
