#!/usr/bin/env python
#
# Test 1D FFT and IFFT calculations.
#
# Hazen 07/19
#
import numpy
import pyopencl as cl

import pyOpenCLNCS
import pyOpenCLNCS.py_ref as pyRef

kernel_code = """
__kernel void fft4_test(__global float4 *x_r, __global float4 *x_c, __global float4 *y_r, __global float4 *y_c) 
{
    float4 r;
    float4 c;
    
    fft4(x_r[0], x_c[0], &r, &c);
    
    y_r[0] = r;
    y_c[0] = c;
}

__kernel void ifft4_test(__global float4 *x_r, 
                         __global float4 *x_c, 
                         __global float4 *y_r, 
                         __global float4 *y_c) 
{
    float4 r;
    float4 c;
    
    ifft4(x_r[0], x_c[0], &r, &c);
    
    y_r[0] = r;
    y_c[0] = c;
}

__kernel void fft8_test(__global float4 *x_r, __global float4 *x_c, __global float4 *y_r, __global float4 *y_c) 
{
    float4 t1_r[2];
    float4 t1_c[2];
    float4 t2_r[2];
    float4 t2_c[2];
    
    t1_r[0] = x_r[0];
    t1_c[0] = x_c[0];
    t1_r[1] = x_r[1];
    t1_c[1] = x_c[1];
    
    fft8(t1_r, t1_c, t2_r, t2_c);
    
    y_r[0] = t2_r[0];
    y_c[0] = t2_c[0];
    y_r[1] = t2_r[1];
    y_c[1] = t2_c[1];
}

__kernel void ifft8_test(__global float4 *x_r, 
                         __global float4 *x_c, 
                         __global float4 *y_r, 
                         __global float4 *y_c) 
{
    float4 t1_r[2];
    float4 t1_c[2];
    float4 t2_r[2];
    float4 t2_c[2];
    
    t1_r[0] = x_r[0];
    t1_c[0] = x_c[0];
    t1_r[1] = x_r[1];
    t1_c[1] = x_c[1];
    
    ifft8(t1_r, t1_c, t2_r, t2_c);
    
    y_r[0] = t2_r[0];
    y_c[0] = t2_c[0];
    y_r[1] = t2_r[1];
    y_c[1] = t2_c[1];
}

__kernel void fft16_test(__global float4 *x_r, __global float4 *x_c, __global float4 *y_r, __global float4 *y_c) 
{
    float4 t1_r[4];
    float4 t1_c[4];
    float4 t2_r[4];
    float4 t2_c[4];
    
    for(int i=0; i<4; i++){
        t1_r[i] = x_r[i];
        t1_c[i] = x_c[i];
    }
    
    fft16(t1_r, t1_c, t2_r, t2_c);
    
    for(int i=0; i<4; i++){
        y_r[i] = t2_r[i];
        y_c[i] = t2_c[i];
    }
}

__kernel void ifft16_test(__global float4 *x_r, 
                          __global float4 *x_c, 
                          __global float4 *y_r, 
                          __global float4 *y_c) 
{
    float4 t1_r[4];
    float4 t1_c[4];
    float4 t2_r[4];
    float4 t2_c[4];
    
    for(int i=0; i<4; i++){
        t1_r[i] = x_r[i];
        t1_c[i] = x_c[i];
    }
    
    ifft16(t1_r, t1_c, t2_r, t2_c);
    
    for(int i=0; i<4; i++){
        y_r[i] = t2_r[i];
        y_c[i] = t2_c[i];
    }
}

__kernel void fft_16x16_test(__global float4 *x_r, 
                             __global float4 *x_c, 
                             __global float4 *y_r, 
                             __global float4 *y_c) 
{
    int lid = get_local_id(0);
    int i = lid*4;
    int j;

    __local float4 t1_r[PSIZE];
    __local float4 t1_c[PSIZE];
    __local float4 t2_r[PSIZE];
    __local float4 t2_c[PSIZE];
    
    for(j=0; j<4; j++){
        t1_r[i+j] = x_r[i+j];
        t1_c[i+j] = x_c[i+j];
    }
    
    fft_16x16_wg16(t1_r, t1_c, t2_r, t2_c, lid);
     
    for(j=0; j<4; j++){
        y_r[i+j] = t2_r[i+j];
        y_c[i+j] = t2_c[i+j];
    }
}

__kernel void ifft_16x16_test(__global float4 *x_r, 
                              __global float4 *x_c, 
                              __global float4 *y_r, 
                              __global float4 *y_c) 
{
    int lid = get_local_id(0);
    int i = lid*4;
    int j;

    __local float4 t1_r[PSIZE];
    __local float4 t1_c[PSIZE];
    __local float4 t2_r[PSIZE];
    __local float4 t2_c[PSIZE];
    
    for(j=0; j<4; j++){
        t1_r[i+j] = x_r[i+j];
        t1_c[i+j] = x_c[i+j];
    }
    
    ifft_16x16_wg16(t1_r, t1_c, t2_r, t2_c, lid);
     
    for(j=0; j<4; j++){
        y_r[i+j] = t2_r[i+j];
        y_c[i+j] = t2_c[i+j];
    }
}

__kernel void fft_16x16_inplace_test(__global float4 *x_r,
                                     __global float4 *x_c, 
                                     __global float4 *y_r,
                                     __global float4 *y_c) 
{
    int lid = get_local_id(0);
    int i = lid*4;
    int j;

    __local float4 t1_r[PSIZE];
    __local float4 t1_c[PSIZE];
    
    for(j=0; j<4; j++){
        t1_r[i+j] = x_r[i+j];
        t1_c[i+j] = x_c[i+j];
    }
    
    fft_16x16_wg16(t1_r, t1_c, t1_r, t1_c, lid);
     
    for(j=0; j<4; j++){
        y_r[i+j] = t1_r[i+j];
        y_c[i+j] = t1_c[i+j];
    }
}

__kernel void ifft_16x16_inplace_test(__global float4 *x_r,
                                      __global float4 *x_c, 
                                      __global float4 *y_r,
                                      __global float4 *y_c) 
{
    int lid = get_local_id(0);
    int i = lid*4;
    int j;

    __local float4 t1_r[PSIZE];
    __local float4 t1_c[PSIZE];
    
    for(j=0; j<4; j++){
        t1_r[i+j] = x_r[i+j];
        t1_c[i+j] = x_c[i+j];
    }
    
    ifft_16x16_wg16(t1_r, t1_c, t1_r, t1_c, lid);
     
    for(j=0; j<4; j++){
        y_r[i+j] = t1_r[i+j];
        y_c[i+j] = t1_c[i+j];
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


def npts_fft(kernel, py_fft, n_pts, g_size = (1,), l_size = (1,)):
   x_r = numpy.random.uniform(size = n_pts).astype(dtype = numpy.float32)
   x_c = numpy.random.uniform(size = n_pts).astype(dtype = numpy.float32)

   y_r = numpy.zeros_like(x_r)
   y_c = numpy.zeros_like(x_c)

   x_r_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = x_r)
   x_c_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = x_c)
    
   y_r_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = y_r)
   y_c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = y_c)
    
   kernel(queue, g_size, l_size, x_r_buffer, x_c_buffer, y_r_buffer, y_c_buffer)
   cl.enqueue_copy(queue, y_r, y_r_buffer).wait()
   cl.enqueue_copy(queue, y_c, y_c_buffer).wait()
   queue.finish()

   x_fft = py_fft(x_r + 1j * x_c)
   
   assert(numpy.allclose(numpy.real(x_fft), y_r, atol = 1.0e-5))
   assert(numpy.allclose(numpy.imag(x_fft), y_c, atol = 1.0e-5))
   

def test_fft_4():
   npts_fft(program.fft4_test, numpy.fft.fft, 4)

def test_fft_8():
   npts_fft(program.fft8_test, numpy.fft.fft, 8)

def test_fft_16():
   npts_fft(program.fft16_test, numpy.fft.fft, 16)

def test_fft_16x16():
   npts_fft(program.fft_16x16_test, numpy.fft.fft2, (16,16), g_size = (16,), l_size = (16,))

def test_fft_16x16_inplace():
   npts_fft(program.fft_16x16_inplace_test, numpy.fft.fft2, (16,16), g_size = (16,), l_size = (16,))

def test_ifft_4():
   npts_fft(program.ifft4_test, numpy.fft.ifft, 4)

def test_ifft_8():
   npts_fft(program.ifft8_test, numpy.fft.ifft, 8)

def test_ifft_16():
   npts_fft(program.ifft16_test, numpy.fft.ifft, 16)

def test_ifft_16x16():
   npts_fft(program.ifft_16x16_test, numpy.fft.ifft2, (16,16), g_size = (16,), l_size = (16,))

def test_ifft_16x16_inplace():
   npts_fft(program.ifft_16x16_inplace_test, numpy.fft.ifft2, (16,16), g_size = (16,), l_size = (16,))

def test_py_ref_fft_16x16(): 
   x_r = numpy.random.uniform(size = 256).astype(dtype = numpy.float32)
   x_c = numpy.random.uniform(size = 256).astype(dtype = numpy.float32)
                              
   y_r = numpy.zeros_like(x_r)
   y_c = numpy.zeros_like(x_c)
    
   y_r_c = numpy.zeros_like(x_r)
   y_c_c = numpy.zeros_like(x_c)
    
   x_r_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = x_r)
   x_c_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = x_c)
    
   y_r_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = y_r)
   y_c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = y_c)
    
   program.fft_16x16_test(queue, (16,), (16,), x_r_buffer, x_c_buffer, y_r_buffer, y_c_buffer)
   cl.enqueue_copy(queue, y_r, y_r_buffer).wait()
   cl.enqueue_copy(queue, y_c, y_c_buffer).wait()
   queue.finish()

   pyRef.fft_16x16(x_r, x_c, y_r_c, y_c_c)
   
   assert(numpy.allclose(y_r, y_r_c, atol = 1.0e-5))
   assert(numpy.allclose(y_c, y_c_c, atol = 1.0e-5))
