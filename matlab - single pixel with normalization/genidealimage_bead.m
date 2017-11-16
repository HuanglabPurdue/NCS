% GENIDEALIMAGE_bead   Simulate normalized ideal image with bead
% 
% SYNOPSIS:
%   [normimg, histimg, normimgL] = genidealimage_bead(imgsz,beadN,Pixelsize,SRzoom,NA,Lambda)
%
% INPUTS:
%   imgsz
%       Size of normalized ideal image
%   beadN
%       number of beads in the image
%   Pixelsize
%       The pixel size on the sample plane, unit is micron
%   SRzoom
%       Subsampling factor of each pixel
%   NA
%       Numerical aperture of the objective lens
%   Lambda
%       Emission wavelength of the sample, unit is micron
%
% OUTPUTS:
%   normimg
%       normalized ideal image, size is equal to the input image size,imgsz
%   histimg 
%       a binary image, represent the object to be imaged, size is equal to imgsz x SRzoom
%   normimgL 
%       normalized ideal image with finer pixel size, size is equal to imgsz x SRzoom
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, November 2017

function [normimg, histimg, normimgL] = genidealimage_bead(imgsz,beadN,Pixelsize,SRzoom,NA,Lambda)
sz = imgsz*SRzoom;
% generate x,y coordinates of beads from a uniform distribution
xtmp = rand(1,beadN); 
ytmp = rand(1,beadN); 
% scale the x,y coordinates to be within the size defined by sz
xsz = max(xtmp(:)); 
ysz = max(ytmp(:));
scale1 = max([ysz,xsz])/sz;
xco1 = xtmp./scale1; 
yco1 = ytmp./scale1;
histimg = SRhist(sz,sz,xco1,yco1); 
% generate normalized ideal image
PSFn = genPSFparam(sz,Pixelsize/SRzoom,NA,Lambda);
normimgL = abs(conv2(double(histimg),double(PSFn),'same'));
if SRzoom > 1
    [imgbin] = binimage(normimgL,SRzoom);
    normimg = imgbin;
else
    normimg = double(normimgL);
end


