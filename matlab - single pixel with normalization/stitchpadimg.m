% STITCHPADIMG   Stitch input segment that were previous padded using segpadimg
% 
% SYNOPSIS:
%   [imgstitch] = stitchpadimg(imgseg)
%
% INPUTS:
%   imgseg
%       Segmented images after padding
%
% OUTPUTS:
%   imgstitch
%       Stitched image
%
% NOTES:
%   If the input segmented image stack has a size of R x R x N, then the out put
%   image will have a size of (R-2)*sqrt(N) x (R-2)*sqrt(N).
%   see also segpadimg
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, April 2017

function [imgstitch] = stitchpadimg(imgseg)
Ns = sqrt(size(imgseg,3));
tmp1 = [];
tmp2 = [];
for ii = 1:Ns*Ns
    tmp1 = cat(2,tmp1,imgseg(2:end-1,2:end-1,ii));
    if mod(ii,Ns) == 0
        tmp2 = cat(1,tmp2,tmp1);
        tmp1 = [];
    end
end
imgstitch = tmp2';
end