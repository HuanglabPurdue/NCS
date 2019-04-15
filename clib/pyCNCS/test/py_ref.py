#!/usr/bin/env python
"""
Python reference versions for testing.

Hazen 04/19
"""
import numpy

def calcLogLikelihood(u, data, gamma):
    return numpy.sum(u-(data+gamma)*numpy.log(u+gamma))

def calcNoiseContribution(u, otfmask):
    normf = u.shape[0]
    tmp = numpy.fft.fftshift(numpy.fft.fft2(u))
    tmp = numpy.abs(tmp)/normf
    tmp = (tmp*otfmask)**2
    return numpy.sum(tmp)
