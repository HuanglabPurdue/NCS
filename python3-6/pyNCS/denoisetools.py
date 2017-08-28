#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:15:56 2017

@author: shengliu
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal as sig
import scipy.optimize as optimize
import scipy.fftpack as ft
import time
import multiprocessing as mp
#import pyfftw as pft

def segpadimg(img,R1):
    R = img.shape[0]
    Ns = R//R1
    ims0 = img[:,0:R1+1]
    for ii in np.arange(1,Ns-1,1):
        tmp = img[:,ii*R1-1:(ii+1)*R1+1] 
        ims0 = np.concatenate((ims0,tmp),axis=1)
    tmp = img[:,(Ns-1)*R1-1:]
    ims0 = np.concatenate((ims0,tmp),axis=1)
    
    ims1 = ims0[0:R1+1,:]
    for ii in np.arange(1,Ns-1,1):
        tmp = ims0[ii*R1-1:(ii+1)*R1+1,:]
        ims1 = np.concatenate((ims1,tmp),axis=0)
    tmp = ims0[(Ns-1)*R1-1:,:]
    ims1 = np.concatenate((ims1,tmp),axis=0)
    ims2 = np.lib.pad(ims1,((1,1),(1,1)),'edge')
    imgsegs = segimg(ims2,R1+2)
    return imgsegs

def segimg(img,R1):
    R = img.shape[0]
    Ns = R//R1
    imgsegs = np.zeros((Ns*Ns,R1,R1))
    for ii in np.arange(0,Ns,1):
        tmp = img[:,ii*R1:(ii+1)*R1]
        imgsegs[ii*Ns:(ii+1)*Ns,:,:] = tmp.reshape((Ns,R1,R1))
    return imgsegs

def padedge(ims,p,axis):
    R = ims.shape[0]
    ims1 = ims.swapaxes(axis,0)
    a = ims1[0]
    b = ims1[-1]
    edge1 = np.zeros((p,R))
    edge2 = np.zeros((p,R))
    for ii in range(p):
        edge1[-(ii)] = (ii+1)*(b-a)/(2*p+1)+a
        edge2[ii] = (ii+1+p)*(b-a)/(2*p+1)+a
    ims2 = np.vstack((edge1,ims1,edge2))
    imspd = ims2.swapaxes(axis,0)
    return imspd
    
def interpad(ims, p):
    ims1 = padedge(ims,p,axis = 0)
    imspd = padedge(ims1,p,axis = 1)
    return imspd

def stitchpadimg(imgseg):
    N = imgseg.shape[0]
    R1 = int(np.sqrt(N)) 
    a = []
    b = []
    for ii in range(N):
        a.append(imgseg[ii,1:-1,1:-1])
        if np.mod(ii+1,R1) == 0:
            b.append(np.vstack(a))
            a = []
    imgstitch = np.hstack(b)
    return imgstitch

def binimage(imgin, ibin):
    sz = imgin.shape[0]   
    R = sz//ibin
    imgsegs = segimg(imgin, ibin)
    imgvec = imgsegs.sum(axis=1).sum(axis=1)
    imgbin = np.reshape(imgvec, (R, R))
    return imgbin

def genkspace(R,pixelsize):
    X,Y = np.meshgrid(np.arange(-R/2,R/2,1),np.arange(-R/2,R/2,1))
    Zo = np.sqrt(X**2+Y**2)
    scale = R*pixelsize
    kr = Zo/scale
    return kr

def genpsfparam(R,pixelsize,NA,Lambda):
    kr = genkspace(R,pixelsize)
    freqmax = NA/Lambda
    pupil  = kr
    pupil[pupil>freqmax] = 0
    psfA = ft.fftshift(ft.fft2(pupil))
    psf = psfA*psfA.conj()
    otf = ft.fftshift(ft.ifft2(psf))
    PSF = psf.real
    OTF = np.abs(otf)
    PSFn = PSF/PSF.sum()
    OTFn = OTF/OTF.max()
    return PSFn,pupil,OTFn
    #return PSFn
    
def SRhist(xsz,ysz,x,y):
    N = x.shape[0]
    frm = 10000
    Ns = N//frm+1
    histim = np.zeros([xsz,ysz])
    for ss in range(Ns):
        st = ss*frm
        if ss == Ns-1:
            ed = N
        else:
            ed = (ss+1)*frm
        tmpx = np.floor(x[st:ed])
        tmpy = np.floor(y[st:ed])
        tmpx = tmpx.astype(int)
        tmpy = tmpy.astype(int)
        mask = (tmpx<xsz) & (tmpy<ysz) & (tmpx>=0) & (tmpy>=0)
        currx = tmpx[mask]
        curry = tmpy[mask]
        idx = np.vstack([currx,curry])
        idx = idx.transpose()
        idx = idx.tolist()
        
        histim[currx,curry] = 1
        x0,y0 = histim.nonzero() # find non zero indices
        idx0 = np.argwhere(histim) # find non zero coordinates pair 
        N0 = idx0.shape[0]
        idx0 = idx0.tolist()
        for nn in range(N0):
            histim[x0[nn],y0[nn]] += idx.count(idx0[nn])
            
    return histim    
        
def genidealimage(R,pixelsize,zoom,NA,Lambda,fpath):
    sz = R*zoom
    cc = sz//2
    Ri = zoom*5//2
    
    #fmat = h5py.loadmat(fpath)
    #varname = list(fmat.keys())
    
    fmat = sio.loadmat(fpath)
    cor = fmat['random_wlc']
    xcor = cor[:,0,:]
    ycor = cor[:,1,:]
    xcor = xcor.ravel()
    ycor = ycor.ravel()
    
    xco = np.round(xcor - xcor.min())
    yco = np.round(ycor - ycor.min())
    xsz = xco.max()
    ysz = yco.max()    
    scale = np.max([xsz,ysz])/sz
    xs = xco/scale
    ys = yco/scale
    histim = SRhist(sz,sz,xs,ys)  
    histim[histim>1] = 1
    res = genpsfparam(sz,pixelsize/zoom,NA,Lambda)
    PSFn = res[0]
    kernel = PSFn[cc-Ri:cc+Ri,cc-Ri:cc+Ri]
    normimgL = sig.fftconvolve(histim,kernel,mode='same')
    normimgL = np.abs(normimgL.transpose())
    
    if zoom>1:
        imgbin = binimage(normimgL,zoom)
        normimg = imgbin/zoom
    else:
        normimg = normimgL
        
    return normimg,kernel

def cropimage(ims,R,startx,starty):
    roi = ims[starty:starty+R,startx:startx+R]
    return roi

def gennoisemap(R,fpath):
    
    #fmat = h5py.loadmat(fpath)
    #varname = list(fmat.keys())
    fmat = sio.loadmat(fpath)
    tmpvar = fmat['ccdvar']
    tmpgain = fmat['gain']
    startx = 92
    starty = 44    
    varsub = cropimage(tmpvar,R,startx,starty)
    gainsub = cropimage(tmpgain,R,startx,starty)
    return varsub,gainsub
       
def addnoise(varmap,gainmap,normimg,I,bg,offset):
    R = normimg.shape[0]
    idealimg = np.abs(normimg)*I+bg
    poissonimg = np.random.poisson(idealimg)
    scmosimg = poissonimg*gainmap + np.sqrt(varmap)*np.random.randn(R,R)
    scmosimg += offset
    return scmosimg,poissonimg 
    
def gendatastack(normimg,varmap,gainmap,I,bg,offset,N):
    R = normimg.shape[0]
    ims = np.zeros([N,R,R])
    imsp = np.zeros([N,R,R])
    imsd = np.zeros([N,R,R])
    for ii in range(N):
        noiseimg = addnoise(varmap,gainmap,normimg,I,bg,offset)
        ims[ii] = noiseimg[0]
        imsp[ii] = noiseimg[1]
        imsd[ii] = (noiseimg[0]-offset)/gainmap
    imso = normimg*I+bg
    imsd[imsd<=0] = 1e-6
    return ims,imsd,imsp,imso

class filters(object):
 
    def pureN(self):
        beta = 0.2
        T = (1-beta)*self.Lambda/4/self.NA
        return beta,T
    def adjustable(self,w,h,kmax):
        w0 = w*self.NA/self.Lambda
        beta = np.pi/2*(kmax/w0-1)/(np.arccos(1-2*h)+np.pi/2*(kmax/w0-1))
        T = (1-beta)/w0/2
        return beta,T
    def weighted(self):
        beta = 1
        T = self.Lambda/4/self.NA/1.4
        return beta,T

def genfilter(R,pixelsize,NA,Lambda,Type='OTFweighted',w=1,h=0.7):
    kr = genkspace(R,pixelsize)
    kmax = 1/np.sqrt(2)/pixelsize
    myfilter = filters()
    myfilter.NA = NA
    myfilter.Lambda = Lambda
    filtertype = {"pureN":myfilter.pureN(),
                  "adjustable":myfilter.adjustable(w,h,kmax),
                  "OTFweighted":myfilter.weighted()}
    param = filtertype[Type]
    beta = param[0]
    T = param[1]
    rcfilter = 1/2.0*(1+np.cos(np.pi*T/beta*(kr-(1-beta)/2/T)))
    mask1 = kr<(1-beta)/2/T
    rcfilter[mask1] = 1
    mask2 = kr>(1+beta)/2/T
    rcfilter[mask2] = 0
    rcfilter = 1-rcfilter
    return rcfilter

def calcost(u,data,var,gain,otfmask,alpha):
    u = u.reshape(data.shape)
    noisepart = calnoisecontri(u,otfmask)
    gamma = var/gain/gain
    LL = u-(data+gamma)*np.log(u+gamma)
    likelihood = LL.sum()
    fcost = likelihood+alpha*noisepart
    return fcost

def calnoisecontri(u,otfmask):
    normf = u.shape[0]
    ims1 = ft.fftshift(ft.fft2(u))    
    #ims1 = sig.fftpack.fftshift(sig.fftpack.fft2(u))    
    ims1 = np.abs(ims1)/normf
    ims2 = (ims1*otfmask)**2
    noisepart = ims2.sum()    
    return noisepart

def segoptim(u0seg,varseg,gainseg,otfmask,alpha,iterationN,ind):
    u0i = u0seg[ind]
    vari = varseg[ind]
    gaini = gainseg[ind]
    #sig.fftpack = pft.interfaces.scipy_fftpack   
    #pft.interfaces.cache.enable() 
    opts = {'disp':False,'maxiter':iterationN}
    outi = optimize.minimize(calcost,u0i,args=(u0i,vari,gaini,otfmask,alpha),method='L-BFGS-B',options=opts)
    outix = outi.x.reshape(u0i.shape)
    return outix
    

def optimf(u0,varseg,gainseg,otfmask,Rs,R,alpha,iterationN):
    Ns = R//Rs
    u0seg = segpadimg(u0,Rs)
    useg = np.zeros(u0seg.shape)#
    t = time.time()
    pool = mp.Pool(processes=8)
    results = [pool.apply_async(segoptim, args=(u0seg,varseg,gainseg,otfmask,alpha,iterationN,ind)) for ind in range(Ns*Ns)]
    outi = [p.get() for p in results]
    useg = np.array(outi)
    pool.close()
    pool.join()
    elapsed = time.time()-t
    print('Elapsed time for noise reduction:', elapsed)
    out = stitchpadimg(useg)
    out[out<0] = 1e-6     
    return out

def reducenoise(Rs,imsd,varmap,gainmap,R,pixelsize,NA,Lambda,alpha,iterationN,Type='OTFweighted',w=1,h=0.7):
    fsz = Rs+2  
    assert imsd.ndim==3, "imsd should be a 3D matrix"
    N = imsd.shape[0]
    outL = np.zeros(imsd.shape)
    rcfilter = genfilter(fsz,pixelsize,NA,Lambda,Type,w,h)
    if gainmap.ndim == 2:
        varseg = segpadimg(varmap,Rs)
        gainseg = segpadimg(gainmap,Rs)
        for ii in range(N):
            out = optimf(imsd[ii],varseg,gainseg,rcfilter,Rs,R,alpha,iterationN)
            outL[ii] = out
    if gainmap.ndim == 3:
        for ii in range(N):            
            varseg = segpadimg(varmap[ii],Rs)
            gainseg = segpadimg(gainmap[ii],Rs)
            out = optimf(imsd[ii],varseg,gainseg,rcfilter,Rs,R,alpha,iterationN)
            outL[ii] = out
    return outL
    

    
    
    
    