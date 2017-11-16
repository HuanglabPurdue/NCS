% FINDOTFRADIUS  Find the OTF radius from the experimental data. 
%
% SYNOPSIS:
%   [freqmax,otf,ind] = findOTFradius(ims,Pixelsize)
%   
% INPUTS:
%   ims
%       A static sequence of sCMOS frames before any preprocessing, high signal to noise
%       ratio is preferable
%   pixelsize
%       pixel size of the image, unit is micron
%
% OUTPUTS:
%   freqmax
%       The cutoff frequency of the OTF support, unit is NA/Lambda
%   otf
%       The logarithm of the modulus of the Fourier transform of the
%       averaged input data stack
%   ind
%       The pixel index of the OTF boundary from the radial average plot of
%       otf
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, June 2017

function [freqmax,otf,ind] = findOTFradius(ims,pixelsize)
R = size(ims,1);
w = round((R/sqrt(2)-R/2)/3);
imsavg = mean(ims,3);
imsavg = interpad(imsavg,1);
otf = double(log(abs(ft(imsavg))));
rdm = double(radialmean(otf));
L = numel(rdm)-w;
stdval = zeros(1,L);

for ii = 1:L
    vs = ii:numel(rdm)-round(w/3);
    stdval(ii) = std(rdm(vs));
    if numel(vs)<20
       break; 
    end
end

[~,ind] = min(stdval(1:ii-1));
freqmax = ind/R/pixelsize/2;
end
