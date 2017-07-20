% SEGIMG   Segment input image
%
% SYNOPSIS:
%   [imgsegs] = segimg(img,R1)
% 
% INPUTS:
%   img
%       Input image
%   R1
%       Segmentation size, R1 must be satisfy R1 x integer = the size of img
%
% OUTPUTS:
%   imgsegs
%       Segmented images with a size of R1 x R1
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, April 2017

function [imgsegs] = segimg(img,R1)
Ns = size(img,1)/R1;

imgsegs = zeros(R1,R1,Ns*Ns);
for ii = 1:Ns
    tmp = img(:,(ii-1)*R1+1:R1*ii);
    imgsegs(:,:,(ii-1)*Ns+1:Ns*ii) = reshape(tmp',[R1,R1,Ns]);
end

end