#!/usr/bin/env python
#
# Test solver.
#
# Hazen 08/19
#
import numpy
import numpy
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

# python3 and C NCS reference version.
import pyCNCS.ncs_c as ncsC

import pyOpenCLNCS.py_ref as pyRef

import pyCUDANCS

#
# CUDA setup.
#
kernel_code = pyCUDANCS.loadNCSKernel()
mod = SourceModule(kernel_code, **pyCUDANCS.src_module_args)
ncsReduceNoise = mod.get_function("ncsReduceNoise")


alpha = 0.1
n_pts = 16

def test_ncs_noise_reduction_1():

   # Setup
   numpy.random.seed(1)

   data = numpy.random.uniform(low = 10.0, high = 20.0, size = (n_pts, n_pts)).astype(dtype = numpy.float32)
   gamma = numpy.random.uniform(low = 2.0, high = 4.0, size = (n_pts, n_pts)).astype(dtype = numpy.float32)
   otf_mask_shift = pyRef.createOTFMask()

   # CUDA Setup.
   u = numpy.zeros((n_pts, n_pts), dtype = numpy.float32)
   iters = numpy.zeros(1, dtype = numpy.int32)
   status = numpy.zeros(1, dtype = numpy.int32)

   # CUDA noise reduction.
   ncsReduceNoise(drv.In(data),
                  drv.In(gamma),
                  drv.In(otf_mask_shift),
                  drv.Out(u),
                  drv.Out(iters),
                  drv.Out(status),
                  numpy.float32(alpha),
                  block = (16,1,1),
                  grid = (1,1))

   # Python reference version.
   ref_u = numpy.zeros(data.size)
   ref_iters = numpy.zeros_like(iters)
   ref_status = numpy.zeros_like(status)
   
   [py_u_fft_grad_r, py_u_fft_grad_c] = pyRef.createUFFTGrad()
   pyRef.ncsReduceNoise(py_u_fft_grad_r,
                        py_u_fft_grad_c,
                        data,
                        gamma,
                        otf_mask_shift,
                        ref_u,
                        ref_iters,
                        ref_status,
                        numpy.float32(alpha))

   ref_u = numpy.reshape(ref_u, data.shape)
   norm_diff = numpy.max(numpy.abs(u[:,:] - ref_u[:,:]))/numpy.max(ref_u[:,:])
   assert(norm_diff < 1.0e-2), str(norm_diff)

def test_ncs_noise_reduction_2():

   # Setup
   numpy.random.seed(1)
   n_reps = 10

   data = numpy.random.uniform(low = 10.0, high = 20.0, size = (n_reps, n_pts, n_pts)).astype(dtype = numpy.float32)
   gamma = numpy.random.uniform(low = 2.0, high = 4.0, size = (n_reps, n_pts, n_pts)).astype(dtype = numpy.float32)
   otf_mask_shift = pyRef.createOTFMask()

   # CUDA Setup.
   u = numpy.zeros((n_reps, n_pts, n_pts), dtype = numpy.float32)
   iters = numpy.zeros(n_reps, dtype = numpy.int32)
   status = numpy.zeros(n_reps, dtype = numpy.int32)

   # CUDA noise reduction.
   ncsReduceNoise(drv.In(data),
                  drv.In(gamma),
                  drv.In(otf_mask_shift),
                  drv.Out(u),
                  drv.Out(iters),
                  drv.Out(status),
                  numpy.float32(alpha),
                  block = (16,1,1),
                  grid = (n_reps,1))

   # NCSC noise reduction.
   otf_mask = numpy.fft.fftshift(otf_mask_shift.reshape(16, 16))
   
   ref_u = numpy.zeros_like(data)

   ncs_sr = ncsC.NCSCSubRegion(r_size = n_pts)

   for i in range(n_reps):
      ncs_sr.newRegion(data[i,:,:], gamma[i,:,:])
      ncs_sr.setOTFMask(otf_mask)
      ref_u[i,:,:] = ncs_sr.cSolve(alpha, verbose = False)

   ncs_sr.cleanup()

   for i in range(n_reps):
      norm_diff = numpy.max(numpy.abs(u[i,:,:] - ref_u[i,:,:]))/numpy.max(ref_u[i,:,:])
      assert(norm_diff < 1.0e-2), "failed {0:d} {1:.3f}".format(i, norm_diff)
      

if (__name__ == "__main__"):
   test_ncs_noise_reduction_1()
   
