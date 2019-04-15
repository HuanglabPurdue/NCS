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


    
if (__name__ == "__main__"):
    test_sr_1()
    
