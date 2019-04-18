#!/usr/bin/env python
"""
Test NCS sub-region calculations.

Hazen 04/19
"""
import numpy

import pyCNCS.ncs_c as ncsC
import pyCNCS.test.py_ref as pyRef

def test_sr_1():
    """
    Test log likelihood calculation.
    """
    im_size = 16
    ncs_sr = ncsC.NCSCSubRegion(im_size)

    for i in range(10):
        gamma = numpy.random.uniform(low = 2.0, high = 4.0, size = (im_size, im_size))
        image = numpy.random.uniform(low = 0.01, high = 10.0, size = (im_size, im_size))
        u = numpy.random.uniform(low = 0.01, high = 10.0, size = (im_size, im_size))

        ncs_sr.newRegion(image, gamma, 1.0)
        ncs_sr.setU(u)
        t1 = ncs_sr.calcLogLikelihood()

        t2 = pyRef.calcLogLikelihood(u, image, gamma)

        assert(numpy.allclose(t1,t2))

    ncs_sr.cleanup()

def test_sr_2():
    """
    Test noise calculation.
    """
    im_size = 16
    ncs_sr = ncsC.NCSCSubRegion(im_size)

    for i in range(10):
        otfmask = pyRef.randomOTFMask(im_size)
        u = numpy.random.uniform(low = 0.01, high = 10.0, size = (im_size, im_size))

        ncs_sr.setOTFMask(otfmask)
        ncs_sr.setU(u)
        t1 = ncs_sr.calcNoiseContribution()

        t2 = pyRef.calcNoiseContribution(u, otfmask)

        assert(numpy.allclose(t1,t2))

    ncs_sr.cleanup()

def test_sr_3():
    """
    Test log likelihood gradient calculation.
    """
    im_size = 16
    ncs_sr = ncsC.NCSCSubRegion(im_size)

    for i in range(10):
        gamma = numpy.random.uniform(low = 2.0, high = 4.0, size = (im_size, im_size))
        image = numpy.random.uniform(low = 0.01, high = 10.0, size = (im_size, im_size))
        u = numpy.random.uniform(low = 0.01, high = 10.0, size = (im_size, im_size))

        ncs_sr.newRegion(image, gamma, 1.0)
        ncs_sr.setU(u)
        t1 = ncs_sr.calcLLGradient()

        t2 = pyRef.calcLLGradient(u, image, gamma)

        assert(numpy.allclose(t1,t2,atol = 1.0e-6))

    ncs_sr.cleanup()

def test_sr_4():
    """
    Test noise contribution gradient calculation.
    """
    im_size = 16
    ncs_sr = ncsC.NCSCSubRegion(im_size)

    for i in range(10):
        otfmask = pyRef.randomOTFMask(im_size)
        u = numpy.random.uniform(low = 0.01, high = 10.0, size = (im_size, im_size))

        ncs_sr.setOTFMask(otfmask)
        ncs_sr.setU(u)
        ncs_sr.calcNoiseContribution()
        
        t1 = ncs_sr.calcNCGradient()
        t2 = pyRef.calcNCGradient(u, otfmask)

        assert(numpy.allclose(t1,t2,atol = 1.0e-5))

    ncs_sr.cleanup()

def test_sr_5():
    """
    Test solvers.
    """
    im_size = 16
    ncs_sr = ncsC.NCSCSubRegion(im_size)
    alpha = 0.02
    verbose = False
    
    for i in range(10):
        gamma = numpy.random.uniform(low = 2.0, high = 4.0, size = (im_size, im_size))
        image = numpy.random.uniform(low = 0.01, high = 10.0, size = (im_size, im_size))
        otfmask = pyRef.randomOTFMask(im_size)

        ncs_sr.newRegion(image, gamma, alpha)
        ncs_sr.setOTFMask(otfmask)

        im1 = pyRef.ncsSolve(image, gamma, otfmask, alpha, verbose = verbose)
        im2 = ncs_sr.pySolve(alpha, verbose = verbose)
        im3 = ncs_sr.pySolveGradient(alpha, verbose = verbose)
        im4 = ncs_sr.cSolve(alpha, verbose = verbose)

        if verbose:
            print(numpy.max(numpy.abs(im1-im2)),
                  numpy.max(numpy.abs(im1-im3)),
                  numpy.max(numpy.abs(im1-im4)))

        assert(numpy.allclose(im1,im2,atol = 1.0e-2))
        assert(numpy.allclose(im1,im3,atol = 1.0e-2))
        assert(numpy.allclose(im1,im4,atol = 1.0e-2))

    ncs_sr.cleanup()

    
if (__name__ == "__main__"):
    test_sr_5()
    
