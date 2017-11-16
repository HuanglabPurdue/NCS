% BINIMAGE   Bin input image with a given factor
%
% SYNOPSIS:
%   [imgbin] = binimage(imgin,pn)
%
% INPUTS:
%   imgin
%       Input image
%   bin
%       Bin factor, it is a integer
%
% OUTPUTS:
%   imgbin
%       Output image after binning
%
% NOTES:
%   The size of imgbin = the size of imgin / bin. 
%   The pixel value of imgbin is equal to the sum of the pixel values of
%   imgin within the binned pixel.
%   see also unbinimage
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, April 2017

function [imgbin] = binimage(imgin,bin)
sz = size(imgin,1);
img = double(imgin);
Imgsz = sz/bin;

[imgsegs] = segimg(img,bin);
imgvec = sum(sum(imgsegs,1),2);
imgbin = reshape(imgvec,Imgsz,Imgsz);
