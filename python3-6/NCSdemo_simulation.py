
"""
------ Demo code for noise correction algorithm for sCMOS camera (NCS algorithm) on simulated data------------
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
import matplotlib.pyplot as plt
import time
import pyNCS.denoisetools as ncs


if __name__ == "__main__":
         
    # create normalized ideal image
    fpath1 = r'randwlcposition.mat'
    imgsz = 128
    zoom = 8
    Pixelsize = 0.1
    NA = 1.4
    Lambda = 0.7
    t = time.time()
    res = ncs.genidealimage(imgsz,Pixelsize,zoom,NA,Lambda,fpath1)
    elapsed = time.time()-t
    print('Elapsed time for generating ideal image:', elapsed)
    imso = res[0]
    plt.imshow(imso,cmap=plt.cm.gray)
    
    # select variance map from calibrated map data
    fpath = r'gaincalibration_561_gain.mat'        
    noisemap = ncs.gennoisemap(imgsz,fpath)
    varsub = noisemap[0]*10 # increase the readout noise by 10 to demonstrate the effect of NCS algorithm
    gainsub = noisemap[1]
    
    # generate simulated data
    I = 100
    bg = 10
    offset = 100
    N = 1
    dataimg = ncs.gendatastack(imso,varsub,gainsub,I,bg,offset,N)
    imsd = dataimg[1]

    # generate noise corrected image
    NA = 1.4
    Lambda = 0.7
    Rs = 8
    iterationN = 15
    alpha = 0.1    
    out = ncs.reducenoise(Rs,imsd[0:1],varsub,gainsub,imgsz,Pixelsize,NA,Lambda,alpha,iterationN)

    f,(ax1,ax2) = plt.subplots(1,2,sharey=False)
    ax1.imshow(imsd[0],aspect='equal',cmap=plt.cm.gray)
    ax2.imshow(out[0],aspect ='equal',cmap=plt.cm.gray)
    plt.show()
