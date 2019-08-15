#!/usr/bin/env python
#
# OpenCL Python wrapper.
#
# Hazen 08/19
#

import os

def loadNCSKernel():
    kernel_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ncs.cl")
    with open(kernel_filename) as fp:
        kernel_code = fp.read()

    return kernel_code
