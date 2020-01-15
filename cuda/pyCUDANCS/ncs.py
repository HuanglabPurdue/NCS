#
# Python wrapper for the CUDA NCS kernel.
#
# Hazen 08/19
#
import numpy
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import warnings
import time

import pyCUDANCS

#
# CUDA setup.
#
kernel_code = pyCUDANCS.loadNCSKernel()
mod = SourceModule(kernel_code, **pyCUDANCS.src_module_args)
ncsReduceNoise = mod.get_function("ncsReduceNoise")


class NCSCUDAException(Exception):
   pass


class NCSCUDA(object):

   def __init__(self, strict = True, **kwds):
      super().__init__(**kwds)

      self.size = 16
      self.strict = strict

   def reduceNoise(self, images, alpha, verbose = False):
      """
      Ideally you process lots of images at the same time for 
      optimal GPU utilization.
      
      Note: Any zero or negative values in the image should be 
            set to a small positive value like 1.0.

      images - A list of images to run NCS on (in units of e-).
      alpha - NCS alpha term.
      """

      s_size = self.size - 2
      im0_shape = images[0].shape

      # First, figure out how many sub-regions in total.
      pad_image = numpy.pad(images[0], 1, 'edge')
      pad_gamma = numpy.pad(self.gamma, 1, 'edge').astype(numpy.float32)

      num_sr = 0
      for i in range(0, pad_image.shape[0], s_size):
         for j in range(0, pad_image.shape[1], s_size):
            num_sr += 1

      num_sr = num_sr * len(images)
      if verbose:
         print("Creating", num_sr, "sub-regions.")

      # Now chop up the images into lots of sub-regions.
      data_in = numpy.zeros((num_sr, self.size, self.size), dtype = numpy.float32)
      gamma = numpy.zeros((num_sr, self.size, self.size), dtype = numpy.float32)
      data_out = numpy.zeros((num_sr, self.size, self.size), dtype = numpy.float32)
      iters = numpy.zeros(num_sr, dtype = numpy.int32)
      status = numpy.zeros(num_sr, dtype = numpy.int32)

      # These store where the sub-regions came from.
      im_i = numpy.zeros(num_sr, dtype = numpy.int32)
      im_bx = numpy.zeros(num_sr, dtype = numpy.int32)
      im_ex = numpy.zeros(num_sr, dtype = numpy.int32)
      im_by = numpy.zeros(num_sr, dtype = numpy.int32)
      im_ey = numpy.zeros(num_sr, dtype = numpy.int32)

      counter = 0
      for h in range(len(images)):
         if (images[h].shape[0] != im0_shape[0]) or (images[h].shape[1] != im0_shape[1]):
            raise NCSCUDAException("All images must be the same size!")
            
         pad_image = numpy.pad(images[h], 1, 'edge')
         for i in range(0, pad_image.shape[0], s_size):
            if ((i + self.size) > pad_image.shape[0]):
               bx = pad_image.shape[0] - self.size
            else:
               bx = i
            ex = bx + self.size
        
            for j in range(0, pad_image.shape[1], s_size):
               if ((j + self.size) > pad_image.shape[1]):
                  by = pad_image.shape[1] - self.size
               else:
                  by = j
               ey = by + self.size

               data_in[counter,:,:] = pad_image[bx:ex,by:ey].astype(numpy.float32)
               gamma[counter,:,:] = pad_gamma[bx:ex,by:ey]

               im_i[counter] = h
               im_bx[counter] = bx
               im_ex[counter] = ex
               im_by[counter] = by
               im_ey[counter] = ey

               counter += 1

      assert (counter == num_sr)
      assert (data_in.dtype == numpy.float32)
      assert (gamma.dtype == numpy.float32)
            
      # Run NCS noise reduction kernel on the sub-regions.
      #
      # FIXME: We could probably do a better job measuring the elapsed time.
      #
      start_time = time.time()
      ncsReduceNoise(drv.In(data_in),
                     drv.In(gamma),
                     drv.In(self.otf_mask),
                     drv.Out(data_out),
                     drv.Out(iters),
                     drv.Out(status),
                     numpy.float32(alpha),
                     block = (16,1,1),
                     grid = (num_sr,1))
      e_time = time.time() - start_time
      
      if verbose:
         print("Processed {0:d} sub-regions in {1:.6f} seconds.".format(num_sr, e_time))

      # Check status.
      failures = {}
      if (numpy.count_nonzero(status != 0) > 0):
         n_fails = numpy.count_nonzero(status != 0)
         n_maxp = numpy.count_nonzero(status == -5)
         if (n_maxp != n_fails):
            warnings.warn("Noise reduction failed on {0:d} sub-regions.".format(n_fails))
         
         # Count number of failures of each type.
         for i in range(-5,0):
            nf = numpy.count_nonzero(status == i)
            if (nf != 0):
               failures[i] = nf

      # FIXME: This needs to be kept in sync with the OpenCL.
      failure_types = {-1 : "Unstarted",
                       -2 : "Reached maximum iterations",
                       -3 : "Increasing gradient",
                       -4 : "Reached minimum step size",
                       -5 : "Reached machine precision"}

      if verbose:
         print("Minimum iterations: {0:d}".format(numpy.min(iters)))
         print("Maximum iterations: {0:d}".format(numpy.max(iters)))
         print("Median iterations: {0:.3f}".format(numpy.median(iters)))
         if bool(failures):
            for key in failures.keys():
               print(failures[key], "failures of type '" + failure_types[key] + "'")
         print()

      # Assemble noise corrected images.
      nc_images = []
      cur_i = -1
      cur_image = None
      for i in range(num_sr):
         if (cur_i != im_i[i]):
            cur_i = im_i[i]
            cur_image = numpy.zeros(im0_shape, dtype = numpy.float32)
            nc_images.append(cur_image)
                
         cur_image[im_bx[i]:im_ex[i]-2,im_by[i]:im_ey[i]-2] = data_out[i, 1:-1,1:-1]

      return nc_images
   
   def setGamma(self, gamma):
      """
      The assumption is that this is the same for all the images.
      
      gamma - CMOS variance (in units of e-).
      """
      self.gamma = gamma.astype(numpy.float32)
        
   def setOTFMask(self, otf_mask):
        
      # Checks.
      if self.strict:

         if (otf_mask.shape[0] != otf_mask.shape[1]):
            raise NCSCUDAException("OTF must be square!")
         
         if (otf_mask.size != self.size*self.size):
            raise NCSCUDAException("OTF size must match sub-region size!")

         if not checkOTFMask(otf_mask):
            raise NCSCUDAException("OTF does not have the expected symmetry!")        

      self.otf_mask = numpy.fft.fftshift(otf_mask).astype(numpy.float32)
        

def checkOTFMask(otf_mask):
   """
   Verify that the OTF mask has the correct symmetries.
   """
   otf_mask_fft = numpy.fft.ifft2(numpy.fft.fftshift(otf_mask))
   if (numpy.max(numpy.imag(otf_mask_fft)) > 1.0e-12):
      return False
   else:
      return True
     

def reduceNoise(images, gamma, otf_mask, alpha, strict = True, verbose = False):
   """
   Run NCS on an image using OpenCL.

   Note: All zero and negatives values in the images should
         be replaced with a small positive value like 1.0.
    
   images - The image to run NCS on (in units of e-).
   gamma - CMOS variance (in units of e-).
   otf_mask - 16 x 16 array containing the OTF mask.
   alpha - NCS alpha term.
   """
   ncs = NCSCUDA()
   ncs.setOTFMask(otf_mask)
   ncs.setGamma(gamma)
   return ncs.reduceNoise(images, alpha, verbose = verbose)
