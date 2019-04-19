## NCS C library ##

A C library for NCS, along with a Python 3 wrapper.

In order to make this also work as a Python package the C library is in the pyCNCS directory.

## Compiling the C library ##

The C library is compiled using the SCons build system.

### Linux ###

```
$ cd NCS/clib
$ scons
```

## Dependencies ##

### C ###

* [FFTW3](http://www.fftw.org/)
* [L-BFGS](http://www.chokkan.org/software/liblbfgs/index.html)

### Python 3 ###

* [numpy](http://www.numpy.org/)
* [scons](https://scons.org/)
