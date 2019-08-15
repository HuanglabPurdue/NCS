#
# Python wrapper for the OpenCL NCS kernel.
#
# Hazen 08/19
#
import numpy
import pyopencl as cl
import warnings

import pyOpenCLNCS

#
# OpenCL setup.
#
kernel_code = pyOpenCLNCS.loadNCSKernel()

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


class NCSOpenCLException(Exception):
    pass


class NCSOpenCL(object):

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
                raise NCSOpenCLException("All images must be the same size!")
            
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
            
        # Run OpenCL noise reduction kernel on the sub-regions.
        data_in_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                                   hostbuf = data_in)
        gamma_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                                 hostbuf = gamma)
        otf_mask_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                                    hostbuf = self.otf_mask)
        data_out_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                                    hostbuf = data_out)
        iters_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                                 hostbuf = iters)
        status_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                                  hostbuf = status)

        ev1 = program.ncsReduceNoise(queue, (num_sr,), (1,),
                                     data_in_buffer,
                                     gamma_buffer,
                                     otf_mask_buffer,
                                     data_out_buffer,
                                     iters_buffer,
                                     status_buffer,
                                     numpy.float32(alpha))

        cl.enqueue_copy(queue, data_out, data_out_buffer).wait()
        cl.enqueue_copy(queue, iters, iters_buffer).wait()
        cl.enqueue_copy(queue, status, status_buffer).wait()
        queue.finish()

        if verbose:
            e_time = 1.0e-9*(ev1.profile.end - ev1.profile.start)
            print("Processed {0:d} sub-regions in {1:.6f} seconds.".format(num_sr, e_time))

        # Check status.
        if (numpy.count_nonzero(status != 0) > 0):
            n_fails = numpy.count_nonzero(status != 0)
            n_maxp = numpy.count_nonzero(status == 5)
            warnings.warn("Noise reduction failed on {0:d} sub-regions, {1:d} of fails reached machine precision.".format(n_fails, n_maxp))

        if verbose:
            print("Minimum iterations: {0:d}".format(numpy.min(iters)))
            print("Maximum iterations: {0:d}".format(numpy.max(iters)))
            print("Median iterations: {0:.3f}".format(numpy.median(iters)))
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
                raise NCSOpenCLException("OTF must be square!")
                        
            if (otf_mask.size != self.size*self.size):
                raise NCSOpenCLException("OTF size must match sub-region size!")

            if not checkOTFMask(otf_mask):
                raise NCSOpenCLException("OTF does not have the expected symmetry!")        

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
    ncs = NCSOpenCL()
    ncs.setOTFMask(otf_mask)
    ncs.setGamma(gamma)
    return ncs.reduceNoise(images, alpha, verbose = verbose)
