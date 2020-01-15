#!/usr/bin/env python
"""
Test NCS on an image.

Hazen 08/19
"""
import numpy

# python3 and C NCS.
import pyCNCS.ncs_c as ncsC
import pyCNCS.test.py_ref as pyRef

# python3 and OpenCL 
import pyCUDANCS.ncs as ncsCU

def test_im_1():
    """
    Verify that C and CUDA NCS code returns the same results 
    on a variety of image sizes.
    """
    im_size = 30
    r_size = 16
    alpha = 0.02
    verbose = False

    numpy.random.seed(1)
    gamma = numpy.random.uniform(low = 2.0, high = 4.0, size = (im_size, im_size))
    image = numpy.random.uniform(low = 0.01, high = 10.0, size = (im_size, im_size))
    otfmask = pyRef.randomOTFMask(r_size)
    otfmask_shift = numpy.fft.fftshift(otfmask)

    for ix in range(12):
        for iy in range(12):
            ncs1 = ncsC.cReduceNoise(image[ix:im_size,iy:im_size],
                                     gamma[ix:im_size,iy:im_size],
                                     otfmask_shift,
                                     alpha)

            ncs2 = ncsCU.reduceNoise([image[ix:im_size,iy:im_size]],
                                     gamma[ix:im_size,iy:im_size],
                                     otfmask,
                                     alpha)

            assert(numpy.allclose(ncs1, ncs2, atol = 0.02))

    
if (__name__ == "__main__"):
    test_im_1()
    
