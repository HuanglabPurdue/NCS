% UNBINIMAGE   Unbin input image with a given factor
%
% SYNOPSIS:
%   [imgL] = unbinimage(img,bin)
%
% INPUTS:
%   img
%       Input image
%   bin
%       Bin factor, it is a integer
%
% OUTPUTS:
%   imgL
%       Output image after unbinning
%
% NOTES:
%   The size of imgL = the size of img x bin. The pixel values of imgL
%   within the corresponding pixel (m) of img is equal to the pixel value
%   of m divided by the square of bin.
%   see also binimage
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, April 2017

function [imgL] = unbinimage(img,bin)
R1 = size(img,1);
vec = img(:);
tmp = repmat(vec,1,bin*bin);
tmp1 = reshape(tmp',bin,bin*R1*R1);
szL = bin*R1;
imgL = zeros(szL,szL);
for ii = 1:R1
    imgL(:,1+bin*(ii-1):bin*ii) = tmp1(:,1+szL*(ii-1):ii*szL)'./bin./bin;
end

end

