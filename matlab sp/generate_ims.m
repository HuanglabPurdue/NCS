% GENERATE_IMS   Generate simulated image
%
% SYNOPSIS:
%   [scmosimg,poissonimg,idealimg] = generate_ims(sz, varmap, gainmap, normimg, I, bg, offset)
%
% INPUTS:
%   varmap
%       selected variance map, size is the same as normimg
%   gainmap
%       selected gain map, size is the same as normimg
%   normimg
%       normalized ideal image
%   I
%       total photon count of each emitter on simulated microtubules
%   bg
%       background photon count
%   offset
%       offset ADU level of sCMOS camera
%
% OUTPUTS:
%   scmosimg
%       Simulated image from sCMOS camera
%   poissonimg
%       Simulated image with Poisson noise only
%   idealimg
%       Simulated ideal image with given photon and background count
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, April 2017

function [scmosimg,poissonimg,idealimg] = generate_ims(varmap, gainmap, normimg, I, bg, offset)
sz = size(normimg,1);
% add photon and background
idealimg = abs(normimg).*I+bg; 
% add Poisson noise 
poissonimg = noise(idealimg,'poisson'); 
% add readout noise and offset
scmosimg = poissonimg.*gainmap + randn(sz,sz).*sqrt(varmap); 
scmosimg = double(scmosimg)+offset;


