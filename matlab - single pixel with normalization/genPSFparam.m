% GENPSFPARAM   Generate PSF, pupil function, and OTF with given parameters
% 
% SYNOPSIS:
%   [PSFn,pupil,OTFn] = genPSFparam(PSFsize,Pixelsize,NA,Lambda)
%
% INPUTS:
%   PSFsize
%       Image size of the PSF, pupil function and OTF
%   Pixelsize
%       The pixel size on the sample plane, unit is micron
%   NA
%       Numerical aperture of the objective lens
%   Lambda
%       Emission wavelength of the sample, unit is micron
%
% OUTPUTS:
%   PSFn
%       normalized PSF image
%   pupil
%       pupil function of ideal imaging system
%   OTFn
%       normalized OTF image
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, April 2017

function [PSFn,pupil,OTFn] = genPSFparam(PSFsize,Pixelsize,NA,Lambda)

[X,Y] = meshgrid(-PSFsize/2:PSFsize/2-1,-PSFsize/2:PSFsize/2-1); 
Zo = sqrt(X.^2+Y.^2);
scale = PSFsize*Pixelsize; 
k_r = Zo./scale;
Freq_max = NA/Lambda;           % cutting-off frequency
pupil = k_r < Freq_max;         % pupil function
PSFA = ft(pupil);               % fourier transform of the pupil
PSF = PSFA.*conj(PSFA);
OTF = abs(ift(PSFA.*conj(PSFA)));
% normalize
OTFn = OTF./max(OTF(:));  
PSFn = PSF./sum(PSF(:));
