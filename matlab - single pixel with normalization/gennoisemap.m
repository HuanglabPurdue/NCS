% GENNOISEMAP    Generate variance and gain map from the selected region of the gain calibration data
%
% SYNOPSIS:
%   [varsub,gainsub] = gennoisemap(sz,filename)
%
% INPUTS:
%   sz
%       Size of selected map
%   filename
%       File name of calibrated map from sCMOS gain calibration
%
% NOTES:
%   The gain calibration data should use 'ccdvar' and 'gain' as the variable
%   name of the variance and gain map
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, April 2017

function [varsub,gainsub] = gennoisemap(sz,filename)
varld = load(filename,'ccdvar','gain');
tmpvar = varld.ccdvar; 
tmpgain = varld.gain; 

rectx=100;
recty=28;
submap = tmpvar(recty:recty+sz-1,rectx:rectx+sz-1);
varsub = submap; 
gainsub = tmpgain(recty:recty+sz-1,rectx:rectx+sz-1);
