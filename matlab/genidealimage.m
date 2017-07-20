% GENIDEALIMAGE   Simulate normalized ideal image
% 
% SYNOPSIS:
%   [normimg, histimg, normimgL] = genidealimage(imgsz,Pixelsize,SRzoom,NA,Lambda)
%
% INPUTS:
%   imgsz
%       Size of normalized ideal image
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
% Sheng Liu, April 2017

function [normimg, histimg, normimgL] = genidealimage(imgsz,Pixelsize,SRzoom,NA,Lambda)
sz = imgsz*SRzoom;
% load coordinates of microtubules from WLC model
tmpld = load('randwlcposition.mat','random_wlc');
xtmp = tmpld.random_wlc(:,1,:); % get the x coordinates
ytmp = tmpld.random_wlc(:,2,:); % get the y coordinates
% scale the x,y coordinates to be within the size defined by sz
xco = xtmp-min(xtmp(:));    
xco = round(xco);
yco = ytmp-min(ytmp(:));
yco = round(yco);
xsz = max(xco(:)); 
ysz = max(yco(:));
scale = max([ysz,xsz])/sz;
xs = xco./scale; 
ys = yco./scale;
histimg = SRhist(sz,sz,xs,ys); 
histimg(histimg>1) = 1;
% generate normalized ideal image
PSFn = genPSFparam(sz,Pixelsize/SRzoom,NA,Lambda);
normimgL = abs(conv2(double(histimg),double(PSFn),'same'));

if SRzoom > 1
    [imgbin] = binimage(normimgL,SRzoom);
    normimg = imgbin./SRzoom;
else
    normimg = double(normimgL);
end


