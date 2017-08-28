
"""
Created on Sun Aug 27 17:20:35 2017

@author: shengliu
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pyNCS.denoisetools as ncs


if __name__ == "__main__":
         
# load coordinates of microtubules 

    fpath1 = r'randwlcposition.mat'
    R = 128
    zoom = 8
    pixelsize = 0.1
    NA = 1.4
    Lambda = 0.7
    t = time.time()
    res = ncs.genidealimage(R,pixelsize,zoom,NA,Lambda,fpath1)
    elapsed = time.time()-t
    print('Elapsed time for generating ideal image:', elapsed)
    imso = res[0]
    plt.imshow(imso,cmap=plt.cm.gray)
    

    fpath = r'gaincalibration_561_gain.mat'    
    I = 100
    bg = 10
    offset = 100
    N = 1
    noisemap = ncs.gennoisemap(R,fpath)
    varsub = noisemap[0]*10
    gainsub = noisemap[1]
    dataimg = ncs.gendatastack(imso,varsub,gainsub,I,bg,offset,N)
    imsd = dataimg[1]

    pixelsize = 0.1
    NA = 1.4
    Lambda = 0.7
    Rs = 8
    iterationN = 15
    alpha = 0.1
    
    out = ncs.reducenoise(Rs,imsd[0:1],varsub,gainsub,R,pixelsize,NA,Lambda,alpha,iterationN)

    f,(ax1,ax2) = plt.subplots(1,2,sharey=False)
    ax1.imshow(imsd[0],aspect='equal',cmap=plt.cm.gray)
    ax2.imshow(out[0],aspect ='equal',cmap=plt.cm.gray)
    plt.show()
