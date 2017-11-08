% SEGPADIMG   Segment input image and pad the segmented images
%
% SYNOPSIS:
%   [imgsegs] = segpadimg(img,R1)
%
% INPUTS:
%   img
%       Input image
%   R1
%       Segmentation size, R1 must be satisfy R1 x integer = size of img
%
% OUTPUTS:
%   imgsegs
%       Segmented images after padding, which has a size of (R1+2) x (R1+2)
%
% NOTES:
%   The segmented images are padded on the edge to form an (R1+2) x (R1+2)
%   image. The padded pixels are generated as follows: 1) for non-edge pixels
%   relative to the original image, it is padded with its adjacent pixel in
%   the original image; 2) for edge pixels relative to the original image, it
%   is padded with itself.
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, April 2017

function [imgsegs] = segpadimg(img,R1)
sz = size(img,1);
Ns = sz/R1;
ims0 = img(:,1:R1+1);
for ii = 2:Ns-1
    ims0 = cat(2,ims0, img(:,(ii-1)*R1:ii*R1+1));
end
ims0 = cat(2,ims0,img(:,(Ns-1)*R1:Ns*R1));

ims1 = ims0(1:R1+1,:);
for ii = 2:Ns-1
    ims1 = cat(1,ims1, ims0((ii-1)*R1:ii*R1+1,:));
end
ims1 = cat(1,ims1,ims0((Ns-1)*R1:Ns*R1,:));
ims2 = padarray(ims1,[1,1],'replicate');

imgsegs = segimg(ims2,R1+2);

end