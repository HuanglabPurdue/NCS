% CALCOST   Calculate cost function in noise correction algorithm
% reference: Liu, Sheng, et al., sCMOS noise-correction algorithm for microscopy images, Nature methods(2017)
%
% SYNOPSIS:
%   [fcost,noisepart,likelihood]=calcost(u,data,var,gain,otfmask,alpha)
%
% INPUTS:
%   u
%       Noise corrected image
%   data
%       sCMOS image after offset and gain correction
%   var
%       variance map, size is the same as data
%   gain
%       gain map, size is the same as data
%   otfmask
%       OTF mask used to calculate the noise contribution in NCS algorithm
%   alpha
%       weight factor of noise contribution only
%
% OUTPUTS:
%   fcost
%       value of the cost function
%   noisepart
%       noise-contribution-only part in Fourier space
%   likelihood
%       likelihood of the input data under model u
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, June 2017

function [fcost,noisepart,likelihood]=calcost(u,data,var,gain,otfmask,alpha)
% calculate the noise contribution of the image in its Fourier space
noisepart = calnoisecontri(u,double(otfmask));
% calculate the likelihood of the input data under model u
gamma = var./gain./gain;
likelihood = sum(u(:) - (data(:) + gamma(:)).*log(u(:) + gamma(:)));
% calculate the cost function
fcost = likelihood + alpha*noisepart;

end

function [noisepart] = calnoisecontri(u, otfmask)

% generate FT image of padded image
normf = size(u,1);
Fimg1 = abs(fftshift(fft2(u))./normf);  % normalize based on parseval's theorem
% apply otfmask, extract noise-contribution-only part in Fourier space
Fimg = Fimg1.*otfmask;
noisepart = sum(Fimg(:).^2);  

end

