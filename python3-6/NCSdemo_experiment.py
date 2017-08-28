
"""
Created on Sun Aug 27 17:20:35 2017

@author: shengliu
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pyNCS.denoisetools as ncs

if __name__ == "__main__":
         
    gainfile = r'gaincalibration_561_gain.mat'  
    datafile = r'EB300004.mat'
    fmat_gain = sio.loadmat(gainfile)
    fmat_data = sio.loadmat(datafile)
    ccdvar = fmat_gain['ccdvar']
    gain = fmat_gain['gain']
    ccdoffset = fmat_gain['ccdoffset']
    data = fmat_data['ims']
    R = 128
    rectx = 59
    recty = 58
    
    subims = ncs.cropimage(data,R,rectx,recty)
    subvar = ncs.cropimage(ccdvar,R,rectx,recty)
    suboffset = ncs.cropimage(ccdoffset,R,rectx,recty)
    subgain = ncs.cropimage(gain,R,rectx,recty)
    subgain[subgain<1] = 1
    subims = subims.swapaxes(0,2) # the subims has a shape of [20,R,R], frame dimension is the first dimension
    subims = subims.swapaxes(1,2)
    N = subims.shape[0]
    imsd = (subims-np.tile(suboffset,(N,1,1)))/np.tile(subgain,(N,1,1))
    imsd[imsd<=0] = 1e-6
    
    
    Rs = 8
    pixelsize = 0.091
    Lambda = 0.54
    NA = 1.35
    iterationN = 15
    alpha = 0.2
    
    out = ncs.reducenoise(Rs,imsd[0:1],subvar,subgain,R,pixelsize,NA,Lambda,alpha,iterationN)

    f,(ax1,ax2) = plt.subplots(1,2,sharey=False)
    ax1.imshow(imsd[0],aspect='equal',cmap=plt.cm.gray)
    ax2.imshow(out[0],aspect ='equal',cmap=plt.cm.gray)
    plt.show()
