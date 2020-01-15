#!/usr/bin/env python
#
# Test vector functions.
#
# Hazen 07/19
#
import numpy
import pyopencl as cl

import pyOpenCLNCS
import pyOpenCLNCS.py_ref as pyRef


kernel_code = """

__kernel void veccopy_test(__global float4 *g_v1,
                           __global float4 *g_v2)
{
    int lid = get_local_id(0);
    int i = lid*4;

    __local float4 v1[PSIZE];
    __local float4 v2[PSIZE];
      
    v2[i]   = g_v2[i];
    v2[i+1] = g_v2[i+1];
    v2[i+2] = g_v2[i+2];
    v2[i+3] = g_v2[i+3];
    
    veccopy(v1, v2, lid);

    g_v1[i]   = v1[i];
    g_v1[i+1] = v1[i+1];
    g_v1[i+2] = v1[i+2];
    g_v1[i+3] = v1[i+3];
}

__kernel void vecncopy_test(__global float4 *g_v1,
                            __global float4 *g_v2)
{
    int lid = get_local_id(0);
    int i = lid*4;

    __local float4 v1[PSIZE];
    __local float4 v2[PSIZE];
      
    v2[i]   = g_v2[i];
    v2[i+1] = g_v2[i+1];
    v2[i+2] = g_v2[i+2];
    v2[i+3] = g_v2[i+3];
    
    vecncopy(v1, v2, lid);

    g_v1[i]   = v1[i];
    g_v1[i+1] = v1[i+1];
    g_v1[i+2] = v1[i+2];
    g_v1[i+3] = v1[i+3];
}

__kernel void vecdot_test(__global float4 *g_v1,
                          __global float4 *g_v2,
                          __global float *g_sum)
{
    int lid = get_local_id(0);
    int i = lid*4;

    __local float w1[16];
    __local float4 v1[PSIZE];
    __local float4 v2[PSIZE];
      
    v1[i]   = g_v1[i];
    v1[i+1] = g_v1[i+1];
    v1[i+2] = g_v1[i+2];
    v1[i+3] = g_v1[i+3];

    v2[i]   = g_v2[i];
    v2[i+1] = g_v2[i+1];
    v2[i+2] = g_v2[i+2];
    v2[i+3] = g_v2[i+3];

    vecdot(w1, v1, v2, lid);
    *g_sum = w1[0];
}

__kernel void vecisEqual_test(__global float4 *g_v1,
                              __global float4 *g_v2,
                              __global int *g_eq)
{
    int lid = get_local_id(0);
    int i = lid*4;

    __local int w1[16];
    __local float4 v1[PSIZE];
    __local float4 v2[PSIZE];
      
    v1[i]   = g_v1[i];
    v1[i+1] = g_v1[i+1];
    v1[i+2] = g_v1[i+2];
    v1[i+3] = g_v1[i+3];

    v2[i]   = g_v2[i];
    v2[i+1] = g_v2[i+1];
    v2[i+2] = g_v2[i+2];
    v2[i+3] = g_v2[i+3];
    
    vecisEqual(w1, v1, v2, lid);
    *g_eq = w1[0];
}

__kernel void vecfma_test(__global float4 *g_v1,
                          __global float4 *g_v2,
                          __global float4 *g_v3,
                          float s1)
{
    int lid = get_local_id(0);
    int i = lid*4;

    __local float4 v1[PSIZE];
    __local float4 v2[PSIZE];
    __local float4 v3[PSIZE];
      
    v2[i]   = g_v2[i];
    v2[i+1] = g_v2[i+1];
    v2[i+2] = g_v2[i+2];
    v2[i+3] = g_v2[i+3];

    v3[i]   = g_v3[i];
    v3[i+1] = g_v3[i+1];
    v3[i+2] = g_v3[i+2];
    v3[i+3] = g_v3[i+3];
    
    vecfma(v1, v2, v3, s1, lid);

    g_v1[i]   = v1[i];
    g_v1[i+1] = v1[i+1];
    g_v1[i+2] = v1[i+2];
    g_v1[i+3] = v1[i+3];
}

__kernel void vecfmaInplace_test(__global float4 *g_v1,
                                 __global float4 *g_v2,
                                 float s1)
{
    int lid = get_local_id(0);
    int i = lid*4;

    __local float4 v1[PSIZE];
    __local float4 v2[PSIZE];

    v1[i]   = g_v1[i];
    v1[i+1] = g_v1[i+1];
    v1[i+2] = g_v1[i+2];
    v1[i+3] = g_v1[i+3];

    v2[i]   = g_v2[i];
    v2[i+1] = g_v2[i+1];
    v2[i+2] = g_v2[i+2];
    v2[i+3] = g_v2[i+3];
    
    vecfmaInplace(v1, v2, s1, lid);

    g_v1[i]   = v1[i];
    g_v1[i+1] = v1[i+1];
    g_v1[i+2] = v1[i+2];
    g_v1[i+3] = v1[i+3];
}

__kernel void vecnorm_test(__global float4 *g_v1,
                           __global float *g_norm)
{
    int lid = get_local_id(0);
    int i = lid*4;

    __local float w1[16];
    __local float4 v1[PSIZE];
      
    v1[i]   = g_v1[i];
    v1[i+1] = g_v1[i+1];
    v1[i+2] = g_v1[i+2];
    v1[i+3] = g_v1[i+3];
    
    vecnorm(w1, v1, lid);
    *g_norm = w1[0];    
}

__kernel void vecscaleInplace_test(__global float4 *g_v1,
                                   float scale)
{
    int lid = get_local_id(0);
    int i = lid*4;

    __local float4 v1[PSIZE];
      
    v1[i]   = g_v1[i];
    v1[i+1] = g_v1[i+1];
    v1[i+2] = g_v1[i+2];
    v1[i+3] = g_v1[i+3];

    vecscaleInplace(v1, scale, lid);

    g_v1[i]   = v1[i];
    g_v1[i+1] = v1[i+1];
    g_v1[i+2] = v1[i+2];
    g_v1[i+3] = v1[i+3];
}

__kernel void vecsub_test(__global float4 *g_v1,
                          __global float4 *g_v2,
                          __global float4 *g_v3)
{
    int lid = get_local_id(0);
    int i = lid*4;

    __local float4 v1[PSIZE];
    __local float4 v2[PSIZE];
    __local float4 v3[PSIZE];
      
    v2[i]   = g_v2[i];
    v2[i+1] = g_v2[i+1];
    v2[i+2] = g_v2[i+2];
    v2[i+3] = g_v2[i+3];

    v3[i]   = g_v3[i];
    v3[i+1] = g_v3[i+1];
    v3[i+2] = g_v3[i+2];
    v3[i+3] = g_v3[i+3];
    
    vecsub(v1, v2, v3, lid);

    g_v1[i]   = v1[i];
    g_v1[i+1] = v1[i+1];
    g_v1[i+2] = v1[i+2];
    g_v1[i+3] = v1[i+3];
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

def test_veccopy():
   v1 = numpy.zeros(n_pts, dtype = numpy.float32)
   v2 = numpy.random.uniform(low = 1.0, high = 10.0, size = n_pts).astype(dtype = numpy.float32)

   v1_c = numpy.copy(v1)
   v2_c = numpy.copy(v2)

   v1_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
   v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)

   program.veccopy_test(queue, (16,), (16,), v1_buffer, v2_buffer)
   cl.enqueue_copy(queue, v1, v1_buffer).wait()
   queue.finish()

   assert numpy.allclose(v1, v2)

   pyRef.veccopy(v1_c, v2_c)
   assert numpy.allclose(v1_c, v2_c)
   
def test_vecncopy():
   v1 = numpy.zeros(n_pts, dtype = numpy.float32)
   v2 = numpy.random.uniform(low = 1.0, high = 10.0, size = n_pts).astype(dtype = numpy.float32)

   v1_c = numpy.copy(v1)
   v2_c = numpy.copy(v2)

   v1_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
   v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)

   program.vecncopy_test(queue, (16,), (16,), v1_buffer, v2_buffer)
   cl.enqueue_copy(queue, v1, v1_buffer).wait()
   queue.finish()
   
   assert numpy.allclose(v1, -v2)

   pyRef.vecncopy(v1_c, v2_c)
   assert numpy.allclose(v1_c, -v2_c)

def test_vecdot():
   v1 = numpy.random.uniform(low = 1.0, high = 10.0, size = n_pts).astype(dtype = numpy.float32)
   v2 = numpy.random.uniform(low = 1.0, high = 10.0, size = n_pts).astype(dtype = numpy.float32)
   v3 = numpy.zeros(1).astype(numpy.float32)

   v1_c = numpy.copy(v1)
   v2_c = numpy.copy(v2)

   v1_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
   v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)
   v3_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v3)
   
   program.vecdot_test(queue, (16,), (16,), v1_buffer, v2_buffer, v3_buffer)
   cl.enqueue_copy(queue, v3, v3_buffer).wait()
   queue.finish()

   assert numpy.allclose(v3, numpy.sum(v1*v2))

   v3_c = pyRef.vecdot(v1_c, v2_c)
   assert numpy.allclose(v3, numpy.sum(v1*v2))

def test_vecisEqual_1():
   v1 = numpy.random.uniform(low = 1.0, high = 10.0, size = n_pts).astype(dtype = numpy.float32)
   v2 = numpy.random.uniform(low = 1.0, high = 10.0, size = n_pts).astype(dtype = numpy.float32)
   v3 = numpy.zeros(1).astype(numpy.int32)

   v1_c = numpy.copy(v1)
   v2_c = numpy.copy(v2)

   v1_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
   v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)
   v3_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v3)
   
   program.vecisEqual_test(queue, (16,), (16,), v1_buffer, v2_buffer, v3_buffer)
   cl.enqueue_copy(queue, v3, v3_buffer).wait()
   queue.finish()

   assert (v3[0] == 0)

   v3_c = pyRef.vecisEqual(v1_c, v2_c)
   assert (v3_c == 0)
   
def test_vecisEqual_2():
   v1 = numpy.random.uniform(low = 1.0, high = 10.0, size = n_pts).astype(dtype = numpy.float32)
   v2 = numpy.copy(v1)
   v3 = numpy.zeros(1).astype(numpy.int32)

   v1_c = numpy.copy(v1)
   v2_c = numpy.copy(v2)

   v1_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
   v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)
   v3_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v3)
   
   program.vecisEqual_test(queue, (16,), (16,), v1_buffer, v2_buffer, v3_buffer)
   cl.enqueue_copy(queue, v3, v3_buffer).wait()
   queue.finish()

   assert (v3[0] == 1)

   v3_c = pyRef.vecisEqual(v1_c, v2_c)
   assert (v3_c == 1)
   
def test_vecfma():
   v1 = numpy.zeros(n_pts, dtype = numpy.float32)
   v2 = numpy.random.uniform(low = 1.0, high = 10.0, size = n_pts).astype(dtype = numpy.float32)
   v3 = numpy.random.uniform(low = 1.0, high = 10.0, size = n_pts).astype(dtype = numpy.float32)
   v4 = numpy.float32(2.0)

   v1_c = numpy.copy(v1)
   v2_c = numpy.copy(v2)
   v3_c = numpy.copy(v3)

   v1_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
   v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)
   v3_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v3)

   program.vecfma_test(queue, (16,), (16,), v1_buffer, v2_buffer, v3_buffer, v4)
   cl.enqueue_copy(queue, v1, v1_buffer).wait()
   queue.finish()
   
   assert numpy.allclose(v1, v2*v4 + v3)

   pyRef.vecfma(v1_c, v2_c, v3_c, v4)
   assert numpy.allclose(v1_c, v2_c*v4 + v3_c)

def test_vecfmaInplace():
   v1 = numpy.random.uniform(low = 1.0, high = 10.0, size = n_pts).astype(dtype = numpy.float32)
   v2 = numpy.random.uniform(low = 1.0, high = 10.0, size = n_pts).astype(dtype = numpy.float32)
   v3 = numpy.float32(2.0)

   v1_ref = numpy.copy(v1)
   v1_c = numpy.copy(v1)
   v2_c = numpy.copy(v2)
   
   v1_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
   v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)
   
   program.vecfmaInplace_test(queue, (16,), (16,), v1_buffer, v2_buffer, v3)
   cl.enqueue_copy(queue, v1, v1_buffer).wait()
   queue.finish()
   
   assert numpy.allclose(v1, v2*v3 + v1_ref)

   pyRef.vecfmaInplace(v1_c, v2_c, v3)
   assert numpy.allclose(v1_c, v2_c*v3 + v1_ref)

def test_vecnorm():
   v1 = numpy.random.uniform(low = 1.0, high = 10.0, size = n_pts).astype(dtype = numpy.float32)
   v2 = numpy.zeros(1).astype(numpy.float32)

   v1_c = numpy.copy(v1)
   
   v1_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
   v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)

   program.vecnorm_test(queue, (16,), (16,), v1_buffer, v2_buffer)
   cl.enqueue_copy(queue, v2, v2_buffer).wait()
   queue.finish()
   
   assert numpy.allclose(v2, numpy.linalg.norm(v1))
   
   v2_c = pyRef.vecnorm(v1)
   assert numpy.allclose(v2_c, numpy.linalg.norm(v1))
   
def test_vecscaleInplace():
   v1 = numpy.random.uniform(low = 1.0, high = 10.0, size = n_pts).astype(dtype = numpy.float32)
   v2 = numpy.float32(0.5)

   v1_c = numpy.copy(v1)
   v1_ref = numpy.copy(v1)*v2
   
   v1_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)

   program.vecscaleInplace_test(queue, (16,), (16,), v1_buffer, v2)
   cl.enqueue_copy(queue, v1, v1_buffer).wait()
   queue.finish()
   
   assert numpy.allclose(v1, v1_ref)

   pyRef.vecscaleInplace(v1_c, v2)
   assert numpy.allclose(v1_c, v1_ref)

def test_vecsub():
   v1 = numpy.zeros(n_pts, dtype = numpy.float32)
   v2 = numpy.random.uniform(low = 1.0, high = 10.0, size = n_pts).astype(dtype = numpy.float32)
   v3 = numpy.random.uniform(low = 1.0, high = 10.0, size = n_pts).astype(dtype = numpy.float32)

   v1_c = numpy.copy(v1)
   v2_c = numpy.copy(v2)
   v3_c = numpy.copy(v3)

   v1_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v1)
   v2_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v2)
   v3_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = v3)
   
   program.vecsub_test(queue, (16,), (16,), v1_buffer, v2_buffer, v3_buffer)
   cl.enqueue_copy(queue, v1, v1_buffer).wait()
   queue.finish()

   assert numpy.allclose(v1, v2-v3)

   pyRef.vecsub(v1_c, v2_c, v3_c)
   assert numpy.allclose(v1_c, v2_c-v3_c)
