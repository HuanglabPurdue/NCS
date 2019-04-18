#
# Python binding for the NCS C library.
#
# Hazen 04/19
#

import ctypes
import numpy
from numpy.ctypeslib import ndpointer
import scipy
import scipy.optimize

import pyCNCS.loadclib as loadclib


ncs = loadclib.loadNCSCLibrary()

ncs.ncsSRCalcLLGradient.argtypes = [ctypes.c_void_p,
                                    ndpointer(dtype = numpy.float64)]

ncs.ncsSRCalcLogLikelihood.argtypes = [ctypes.c_void_p]
ncs.ncsSRCalcLogLikelihood.restype = ctypes.c_double

ncs.ncsSRCalcNCGradient.argtypes = [ctypes.c_void_p,
                                    ndpointer(dtype = numpy.float64)]

ncs.ncsSRCalcNoiseContribution.argtypes = [ctypes.c_void_p]
ncs.ncsSRCalcNoiseContribution.restype = ctypes.c_double

ncs.ncsSRCleanup.argtypes = [ctypes.c_void_p]

ncs.ncsSRGetU.argtypes = [ctypes.c_void_p,
                          ndpointer(dtype = numpy.float64)]

ncs.ncsSRInitialize.argtypes = [ctypes.c_int]
ncs.ncsSRInitialize.restype = ctypes.c_void_p

ncs.ncsSRNewRegion.argtypes = [ctypes.c_void_p,
                               ndpointer(dtype = numpy.float64),
                               ndpointer(dtype = numpy.float64),
                               ctypes.c_double]

ncs.ncsSRSetOTFMask.argtypes = [ctypes.c_void_p,
                                ndpointer(dtype = numpy.float64)]

ncs.ncsSRSetU.argtypes = [ctypes.c_void_p,
                          ndpointer(dtype = numpy.float64)]

ncs.ncsSRSolve.argtypes = [ctypes.c_void_p,
                           ctypes.c_double,
                           ctypes.c_int]
ncs.ncsSRSolve.restype = ctypes.c_int


class NCSCException(Exception):
    pass


class NCSCSubRegion(object):

    def __init__(self, r_size = None, strict = True, **kwds):
        super().__init__(**kwds)
        self.alpha = None
        self.r_size = r_size
        self.strict = strict

        if ((r_size%2)!=0):
            raise NCSException("Sub region size must be divisible by 2!")
        
        self.c_ncs = ncs.ncsSRInitialize(r_size)

    def calcCost(self, u):
        """
        This is used by pySolve().
        """
        self.setU(u.reshape(self.image.shape))
        return self.calcLogLikelihood() + self.alpha*self.calcNoiseContribution()

    def calcCostGradient(self, u):
        """
        This is used by pySolve().
        """
        return self.calcLLGradient() + self.alpha*self.calcNCGradient()
    
    def calcLLGradient(self):
        gradient = numpy.zeros(self.r_size*self.r_size, dtype = numpy.float64)
        ncs.ncsSRCalcLLGradient(self.c_ncs, gradient)
        return gradient    

    def calcLogLikelihood(self):
        return ncs.ncsSRCalcLogLikelihood(self.c_ncs)

    def calcNCGradient(self):
        gradient = numpy.zeros(self.r_size*self.r_size, dtype = numpy.float64)
        ncs.ncsSRCalcNCGradient(self.c_ncs, gradient)
        return gradient    

    def calcNoiseContribution(self):
        return ncs.ncsSRCalcNoiseContribution(self.c_ncs)

    def cleanup(self):
        ncs.ncsSRCleanup(self.c_ncs)
        self.c_ncs = None

    def cSolve(self, alpha, verbose = True):
        ret = ncs.ncsSRSolve(self.c_ncs, alpha, verbose)
        if verbose:
            print("L-BFGS method returned {0:d}".format(ret));
        return self.getU()

    def getU(self):
        u = numpy.zeros((self.r_size, self.r_size), dtype = numpy.float64)
        ncs.ncsSRGetU(self.c_ncs, u)
        return u

    def newRegion(self, image, gamma, alpha):
        self.image = image

        # Checks.
        if self.strict:

            if (image.shape[0] != image.shape[1]):
                raise NCSCException("Sub-region image must be square!")
                        
            if (image.size != self.r_size*self.r_size):
                raise NCSCException("Sub-region image size must match sub-region size!")

            if (gamma.shape[0] != gamma.shape[1]):
                raise NCSCException("Sub-region gamma must be square!")
                        
            if (gamma.size != self.r_size*self.r_size):
                raise NCSCException("Sub-region gamma size must match sub-region size!")

        ncs.ncsSRNewRegion(self.c_ncs,
                           numpy.ascontiguousarray(image, dtype = numpy.float64),
                           numpy.ascontiguousarray(gamma, dtype = numpy.float64),
                           alpha)

    def pySolve(self, alpha, verbose = True):
        """
        This is primarily for testing.

        Solve using scipy L-BFGS-B minimizer, without gradient information.
        """
        self.alpha = alpha
        
        opts = {'disp' : verbose, 'maxiter' : 200}
        outi = scipy.optimize.minimize(self.calcCost,
                                       self.image,
                                       method='L-BFGS-B',
                                       options=opts)

        if verbose:
            print("Status", outi.status) 
            print("Message", outi.message)
            print("Iterations", outi.nit)
            
        outix = outi.x.reshape(self.image.shape)
        return outix

    def pySolveGradient(self, alpha, verbose = True):
        """
        This is primarily for testing.

        Solve using scipy L-BFGS-B minimizer, using gradient information.
        """
        self.alpha = alpha
        
        opts = {'disp' : verbose, 'maxiter' : 200}
        outi = scipy.optimize.minimize(self.calcCost,
                                       self.image,
                                       jac = self.calcCostGradient,
                                       method='L-BFGS-B',
                                       options=opts)

        if verbose:
            print("Status", outi.status) 
            print("Message", outi.message)
            print("Iterations", outi.nit)
            
        outix = outi.x.reshape(self.image.shape)
        return outix
        
    def setOTFMask(self, otf_mask):
        
        # Checks.
        if self.strict:

            if (otf_mask.shape[0] != otf_mask.shape[1]):
                raise NCSCException("OTF must be square!")
                        
            if (otf_mask.size != self.r_size*self.r_size):
                raise NCSCException("OTF size must match sub-region size!")

            lr = numpy.fliplr(otf_mask)
            if not numpy.allclose(otf_mask, lr):
                raise NCSCException("OTF must by symmetric!")
            
            ud = numpy.flipud(otf_mask)
            if not numpy.allclose(otf_mask, ud):
                raise NCSCException("OTF must by symmetric!")
            
        tmp = numpy.fft.fftshift(otf_mask)
        ncs.ncsSRSetOTFMask(self.c_ncs,
                            numpy.ascontiguousarray(tmp, dtype = numpy.float64))

    def setU(self, u):
        
        # Checks.
        if self.strict:

            if (u.shape[0] != u.shape[1]):
                raise NCSException("u must be square!")
                        
            if (u.size != self.r_size*self.r_size):
                raise NCSException("u size must match sub-region size!")

        ncs.ncsSRSetU(self.c_ncs,
                      numpy.ascontiguousarray(u, dtype = numpy.float64))

