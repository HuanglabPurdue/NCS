% CROPIMAGE   Crop a square region from the input image
%
% SYNOPSIS:
%   [subroi,rect] = cropimage(ims,sz,rect)
%
% INPUTS:
%   ims
%       Input image
%   sz
%       Size of region of interest, it is a integer
%   rect
%       The left, top, width, height of the selected region
%   
% OUTPUTS:
%   subroi
%       Selected region of interest, the size is sz x sz
%   rect
%       The left, top, width, height of the selected region
%
% NOTES:
%   If the input variable rect is empty, the user is required to select a
%   region from the pop up figure window. Otherwise the function will
%   output the selected region defined in rect
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, April 2017


function [subroi,rect] = cropimage(ims,sz,rect)
if isempty(rect)
    h=dipshow(ims);
    diptruesize(h,200)
    [var,rect] = dipcrop(h);
end
rectx=rect(1,1);
recty=rect(1,2);
subroi = ims(recty:recty+sz-1,rectx:rectx+sz-1,:);
