## NCS on a GPU using OpenCL ##

An OpenCL GPU kernel for NCS.

## Performance ##

Tests were done using the `NCS/opencl/pyOpenCLNCS/profile.py` Python script.
Speedup is relative to the clib version of NCS on the same computer.

* Nvidia Tesla K20Xm - 5x speedup.
* Intel Haswell-ULT Integrated Graphics Controller - 4.2x speedup.
* Nvidia GeForce GT 1030 - 1.2x speedup.

## Example Usage ##

Please see the Jupyter notebooks in the `jupyter_notebooks` folder for
examples of how to use the kernels.

### Jupyter notebook dependencies ###

#### Python 3 ####

* [numpy](http://www.numpy.org/)
* [pyopencl](https://documen.tician.de/pyopencl/)
