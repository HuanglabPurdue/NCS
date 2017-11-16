% INTERPAD   Pad the edge pixels of the input image, in order to reduce the
%            edge effect from FFT
%
% SYNOPSIS:
%   imspd = interpad(ims,sz)
%   
% INPUTS:
%   ims
%       The input image, it is a 2D matrix
%   sz
%       The number of pixels to be pad on each edge pixel
%
% NOTE:
%   The padding scheme is as follows:
%       The size of ims is R x R
%       1) for each row, generate 2*sz number of values between the values
%       of the first and the last pixels using linear interpolation, and
%       insert the first half of the values before the first pixel and the
%       second half of the value after the last pixel, the resulting image
%       has R x R+2*sz pixels; 2) for each column, repeat step 1, the
%       resulting image has R+2*sz x R+2*sz pixels.
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, April 2017
function imspd = interpad(ims,sz)
a = ims(:,1);
b = ims(:,end);
tmp = [];
for ii = 1:sz
    tmp = cat(2,ii*(b-a)/(2*sz+1)+a,tmp);
end
tmp = cat(2,tmp,ims);
for ii = sz+1:2*sz
    tmp = cat(2,tmp,ii*(b-a)/(2*sz+1)+a);
end

a1 = tmp(1,:);
b1 = tmp(end,:);
tmp1 = [];
for ii = 1:sz
    tmp1 = cat(1,ii*(b1-a1)/(2*sz+1)+a1,tmp1);
end
tmp1 = cat(1,tmp1,tmp);
for ii = sz+1:2*sz
    tmp1 = cat(1,tmp1,ii*(b1-a1)/(2*sz+1)+a1);
end

imspd = tmp1;
end