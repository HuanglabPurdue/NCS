#!/usr/bin/env python
"""
Handle loading the NCS C library

Hazen 04/19
"""

import ctypes
import sys
import os
import re

import storm_analysis

def loadNCSCLibrary():
    library_filename = "ncs"

    #
    # This assumes that the C library is one level up from where this file is.
    #
    c_lib_path = os.path.dirname(os.path.abspath(__file__))

    # Windows.
    if (sys.platform == "win32"):

        library_filename += '.dll'

        # Try to load the library without fiddling with the Windows DLL search 
        # first. Do this so that if the user wanted to use their own versions 
        # of FFTW or LAPACK these will get loaded instead of the ones in 
        # storm-analysis.
        try:
            # This suppresses the Windows DLL missing dialog.
            ctypes.windll.kernel32.SetErrorMode(0x0001|0x0002|0x8000)
            return ctypes.cdll.LoadLibrary(os.path.join(c_lib_path, library_filename))
        except WindowsError:
            # Unsuppress the Windows DLL missing dialog.
            ctypes.windll.kernel32.SetErrorMode(0)

            # Push the storm-analysis directory into the DLL search path.
            if (sys.version_info > (3, 0)):
                ctypes.windll.kernel32.SetDllDirectoryW(c_lib_path)
            else:
                ctypes.windll.kernel32.SetDllDirectoryW(unicode(c_lib_path))

            # Try to load the library.
            c_lib = ctypes.cdll.LoadLibrary(os.path.join(c_lib_path, library_filename))

            # Restore the Windows DLL search path.
            ctypes.windll.kernel32.SetDllDirectoryW(None)

            return c_lib

    # OS-X.
    elif (sys.platform == "darwin"):
        library_filename = 'lib' + library_filename
        library_filename += '.dylib'
        return ctypes.cdll.LoadLibrary(os.path.join(c_lib_path, library_filename))
        
    # Linux.
    else:
        library_filename = 'lib' + library_filename
        library_filename += '.so'
        return ctypes.cdll.LoadLibrary(os.path.join(c_lib_path, library_filename))


