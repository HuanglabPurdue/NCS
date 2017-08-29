
"""
------ Demo code for noise correction algorithm for sCMOS camera (NCS algorithm) on experimental data------------
 reference: Liu,Sheng,et al.,sCMOS noise-correction algorithm for microscopy images,Nature Methods 14,760-761(2017)
 software requirement: Python 3.6
(C) Copyright 2017                The Huang Lab
    All rights reserved           Weldon School of Biomedical Engineering
                                  Purdue University
                                  West Lafayette, Indiana
                                  USA
 

@author: Sheng Liu and David A. Miller, August 2017

"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pyNCS.denoisetools as ncs

if __name__ == "__main__":
    
    # select data and gain calibration file     
    gainfile = r'gaincalibration_561_gain.mat'  
    datafile = r'EB300004.mat'
    fmat_gain = sio.loadmat(gainfile)
    fmat_data = sio.loadmat(datafile)
    ccdvar = fmat_gain['ccdvar']
    gain = fmat_gain['gain']
    ccdoffset = fmat_gain['ccdoffset']
    data = fmat_data['ims']
    
    # crop region of interest (ROI)
    imgsz = 128
    rectx = 59
    recty = 58    
    subims = ncs.cropimage(data,imgsz,rectx,recty)
    subvar = ncs.cropimage(ccdvar,imgsz,rectx,recty)
    suboffset = ncs.cropimage(ccdoffset,imgsz,rectx,recty)
    subgain = ncs.cropimage(gain,imgsz,rectx,recty)
    subgain[subgain<1] = 1
    subims = subims.swapaxes(0,2) # the subims has a shape of [20,imgsz,imgsz], frame dimension is the first dimension
    subims = subims.swapaxes(1,2)
    
    # apply gain and offset correction
    N = subims.shape[0]
    imsd = (subims-np.tile(suboffset,(N,1,1)))/np.tile(subgain,(N,1,1))
    imsd[imsd<=0] = 1e-6
    
    # generate noise corrected image
    Rs = 8
    Pixelsize = 0.091
    Lambda = 0.54
    NA = 1.35
    iterationN = 15
    alpha = 0.2    
    out = ncs.reducenoise(Rs,imsd[0:1],subvar,subgain,imgsz,Pixelsize,NA,Lambda,alpha,iterationN)

    f,(ax1,ax2) = plt.subplots(1,2,sharey=False)
    ax1.imshow(imsd[0],aspect='equal',cmap=plt.cm.gray)
    ax2.imshow(out[0],aspect ='equal',cmap=plt.cm.gray)
    plt.show()
