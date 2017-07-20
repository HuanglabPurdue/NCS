% GENMAPS   Generate a gain and a variance stack based on the ADU count of
%           the pixel. This is used for the situation when the sCMOS camera
%           has multiple gain amplifiers, which will result in multiple
%           gain and variance maps based on the pixels' ADU level
%
% SYNOPSIS:
%   [gainstack,varstack] = genmaps(ims,gain,var,lightlevel)
%   
% INPUTS:
%   ims
%       The raw sCMOS frames before any preprocessing
%   gain
%       A stack of gain map(s) from camera calibration
%   var
%       A stack of variance map(s) from camera calibration
%   lightlevel
%       Division point of the light levels, the element number is equal to
%       the third dimension of gain minus 1. For example if gain is consist
%       of two maps, then the lightlevel is a scalar
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, June 2017

function [gainstack,varstack] = genmaps(ims,gain,var,lightlevel)
N = size(ims,3);
NL = size(gain,3);
R = size(ims,1);
gain0 = zeros(R,R);
var0 = zeros(R,R);
gainstack = zeros(R,R,N);
varstack = zeros(R,R,N);
for ii = 1:N
    imsi = ims(:,:,ii);
    gain1 = gain(:,:,1);
    var1 = var(:,:,1);
    mask0 = imsi<lightlevel(1);
    gain0(mask0) = gain1(mask0);
    var0(mask0) = var1(mask0);
    for jj = 1:NL-1
        if  numel(lightlevel)>=jj+1
            mask0 = imsi>=lightlevel(jj)&imsi<lightlevel(jj+1);
            gain1 = gain(:,:,jj+1);
            var1 = var(:,:,jj+1);
            gain0(mask0) = gain1(mask0);
            var0(mask0) = var1(mask0);
        else
            mask0 = imsi>=lightlevel(jj);
            gain1 = gain(:,:,jj+1);
            var1 = var(:,:,jj+1);
            gain0(mask0) = gain1(mask0);
            var0(mask0) = var1(mask0);
        end
    end
    gainstack(:,:,ii) = gain0;
    varstack(:,:,ii) = var0;
end
end