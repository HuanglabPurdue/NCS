#
# Python binding for the NCS C library.
#
# Hazen 04/19
#

import ctypes
import numpy
from numpy.ctypeslib import ndpointer

import pyCNCS.loadclib as loadclib

ncs = loadclib.loadNCSCLibrary()

ncs.ncsSRCalcLogLikelihood.argtypes = [ctypes.c_void_p]
ncs.ncsSRCalcLogLikelihood.restype = ctypes.c_double

ncs.ncsSRCalcNoiseContribution.argtypes = [ctypes.c_void_p]
ncs.ncsSRCalcNoiseContribution.restype = ctypes.c_double

ncs.ncsSRCleanup.argtypes = [ctypes.c_void_p]

ncs.ncsSRInitialize.argtypes = [ctypes.c_int]
ncs.ncsSRInitialize.restype = ctypes.c_void_p

ncs.ncsSRNewRegion.argtypes = [ctypes.c_void_p,
                               ndpointer(dtype = numpy.float64),
                               ndpointer(dtype = numpy.float64),
                               ctypes.c_double]

ncs.ncsSRSetOTF.argtypes = [ctypes.c_void_p,
                            ndpointer(dtype = numpy.float64)]

ncs.ncsSRSetU.argtypes = [ctypes.c_void_p,
                          ndpointer(dtype = numpy.float64)]


class NCSCExecption(Exception):
    pass


class NCSCSubRegion(object):

    def __init__(self, r_size = None, strict = True, **kwds):
        super().__init__(**kwds)
        self.r_size = r_size
        self.strict = strict

        if ((r_size%2)!=0):
            raise NCSException("Sub region size must be divisible by 2!")
        
        self.c_ncs = ncs.ncsSRInitialize(r_size)

    def calcLogLikelihood(self):
        return ncs.ncsSRCalcLogLikelihood(self.c_ncs)

    def calcNoiseContribution(self):
        return ncs.ncsSRCalcNoiseContribution(self.c_ncs)

    def cleanup(self):
        ncs.ncsSRCleanup(self.c_ncs)
        self.c_ncs = None

    def newRegion(self, image, gamma, alpha):

        # Checks.
        if self.strict:

            if (image.shape[0] != image.shape[1]):
                raise NCSException("Sub-region image must be square!")
                        
            if (image.size != self.r_size*self.r_size):
                raise NCSException("Sub-region image size must match sub-region size!")

            if (gamma.shape[0] != gamma.shape[1]):
                raise NCSException("Sub-region gamma must be square!")
                        
            if (gamma.size != self.r_size*self.r_size):
                raise NCSException("Sub-region gamma size must match sub-region size!")

        ncs.ncsSRNewRegion(self.c_ncs,
                           numpy.ascontiguousarray(image, dtype = numpy.float64),
                           numpy.ascontiguousarray(gamma, dtype = numpy.float64),
                           alpha)

    def setOTF(self, otf):
        
        # Checks.
        if self.strict:

            if (otf.shape[0] != otf.shape[1]):
                raise NCSException("OTF must be square!")
                        
            if (otf.size != self.r_size*self.r_size):
                raise NCSException("OTF size must match sub-region size!")

        ncs.ncsSRSetOTF(self.c_ncs,
                        numpy.ascontiguousarray(otf, dtype = numpy.float64))

    def setU(self, u):
        
        # Checks.
        if self.strict:

            if (u.shape[0] != u.shape[1]):
                raise NCSException("u must be square!")
                        
            if (u.size != self.r_size*self.r_size):
                raise NCSException("u size must match sub-region size!")

        ncs.ncsSRSetU(self.c_ncs,
                      numpy.ascontiguousarray(u, dtype = numpy.float64))

