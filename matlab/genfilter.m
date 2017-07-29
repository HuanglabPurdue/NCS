% GENFILTER   Generate OTF mask based on raised cosine filter
% reference: Liu, Sheng, et al., sCMOS noise-correction algorithm for microscopy images, Nature methods(2017)
%
% SYNOPSIS:
%   Default: 
%       [PSFn,pupil,OTFn,rcfilter] = genfilter(imgsz,Pixelsize,NA,Lambda)
%   Noise only:
%       [PSFn,pupil,OTFn,rcfilter] = genfilter(imgsz,Pixelsize,NA,Lambda,'pureN')
%   Adjustable:
%       [PSFn,pupil,OTFn,rcfilter] = genfilter(imgsz,Pixelsize,NA,Lambda,'adjustable',w,h)
%       w is the radius of the OTF mask, unit in NA/Lambda, h is the height
%       of the OTF radius.
%   
% INPUTS:
%   imgsz
%       Image size of the OTF mask
%   Pixelsize
%       The pixel size on the sample plane, unit is micron
%   NA
%       Numerical aperture of the objective lens
%   Lambda
%       Emission wavelength of the sample, unit is micron
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, June 2017

function [rcfilter] = genfilter(imgsz,Pixelsize,NA,Lambda,type,w,h)

[X,Y] = meshgrid(-imgsz/2:imgsz/2-1,-imgsz/2:imgsz/2-1); 
Zo = sqrt(X.^2+Y.^2);
scale = imgsz*Pixelsize; 
k_r = Zo./scale;
kmax = imgsz/sqrt(2)/scale;

if nargin>4
    switch type
        case 'pureN'
            beta = 0.2;
            T = (1-beta)*Lambda/4/NA;
        case 'adjustable'
            w0 = w*NA/Lambda;
            beta = pi/2*(kmax/w0-1)/(acos(1-2*h)+pi/2*(kmax/w0-1));
            T = (1-beta)/w0/2;
    end
else
    T = Lambda/4/NA/1.4;
    beta = 1;
end

rcfilter = 1/2.*(1+cos(pi*T/beta.*(k_r-(1-beta)/2/T)));
mask1 = k_r<(1-beta)/2/T;
rcfilter(mask1) = 1;
mask2 = k_r>(1+beta)/2/T;
rcfilter(mask2) = 0;
rcfilter = 1-rcfilter;
if sum(rcfilter) == 0
        error('NCS:invalidOTFmask','OTF mask error \nThe OTF mask is all zeros, \nplease choose default or adjustable OTF with smaller radius');
end

