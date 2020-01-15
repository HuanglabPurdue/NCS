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
    int lid = get_local_id(0);
    int i = lid*4;

    __local int w1[1];
    __local float w2[ASIZE];
    __local float4 v1[PSIZE];
    __local float4 v2[PSIZE];
    
    for (int j=0; j<4; j++){
        v1[i+j] = g_v1[i+j];
        v2[i+j] = g_v2[i+j];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    converged(w1, w2, v1, v2, lid);
    *g_conv = w1[0];
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
   
   program.converged_test(queue, (16,), (16,), v1_buffer, v2_buffer, v3_buffer)
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
   
   program.converged_test(queue, (16,), (16,), v1_buffer, v2_buffer, v3_buffer)
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
   
   program.converged_test(queue, (16,), (16,), v1_buffer, v2_buffer, v3_buffer)
   cl.enqueue_copy(queue, v3, v3_buffer).wait()
   queue.finish()
   
   assert (v3[0] == 1)

if (__name__ == "__main__"):
   test_converged_1()
   
