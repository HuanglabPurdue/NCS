#!/usr/bin/env python
#
# CUDA Python wrapper.
#
# Hazen 08/19
#

import os

#
# These are the arguments passed to pycude.compiler.SourceModule for
# compilation. They probably need to be adjusted to reflect your
# CUDA installation.
#
src_module_args = {"nvcc" : "/usr/local/cuda-10.1/bin/nvcc",
                   "no_extern_c" : True,
                   "include_dirs" : ["/usr/local/cuda/samples/common/inc"]}

assert os.path.exists(src_module_args["nvcc"]), "CUDA nvcc not found at " + src_module_args["nvcc"] + "."

def loadNCSKernel():
    kernel_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ncs.cu")
    with open(kernel_filename) as fp:
        kernel_code = fp.read()

    return kernel_code
