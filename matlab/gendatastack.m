% GENDATASTACK   Generate simulated data stacks
%
% SYNOPSIS:
%   [ims,imsd,imsp,imso,imsoL] = gendatastack(normimg,varmap,gainmap,I,bg,offset,imgsz,N)
%
% INPUTS:
%   normimg
%       normalized ideal image
%   varmap
%       selected variance map, size is the same as normimg
%   gainmap
%       selected gain map, size is the same as normimg
%   I
%       total photon count of each emitter on simulated microtubules
%   bg
%       background photon count
%   offset
%       offset ADU level of sCMOS camera
%   N
%       Frame number of simulated data stack
%
% OUTPUTS:
%   ims
%       Simulated image from sCMOS camera
%   imsd 
%       Simulated image from sCMOS camera after gain and offset correction
%   imsp
%       Simulated image with Poisson noise only
%   imso
%       Simulated ideal image with given photon and background count
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, April 2017

function [ims,imsd,imsp,imso] = gendatastack(normimg,varmap,gainmap,I,bg,offset,N)
imgsz = size(normimg,1);
ims = zeros(imgsz,imgsz,N);
imsp = zeros(imgsz,imgsz,N);
for ii = 1:N
    [scmosimg,poissonimg] = generate_ims(varmap,gainmap,normimg,I,bg,offset);
    ims(:,:,ii) = scmosimg;
    imsp(:,:,ii) = poissonimg;
end
imso = normimg.*I+bg;
gainL = repmat(gainmap,[1,1,N]);
imsd = (ims-offset)./gainL;
imsd(imsd<=0) = 1e-6;
imsp = double(imsp);
imsp(imsp<=0) = 1e-6;

end