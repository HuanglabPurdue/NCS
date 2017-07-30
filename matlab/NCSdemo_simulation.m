%------ Demo code for noise correction algorithm for sCMOS camera (NCS algorithm) on simulated data------------
% reference: Liu,Sheng,et al.,sCMOS noise-correction algorithm for microscopy images,Nature Methods 14,760-761(2017)
% software requirement: Matlab R2015a or later
%                       Dipimage toolbox 2.7 or later
% system requirement:   CPU Intel Core i7 
%                       32 GB RAM 
% 
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, June 2017

%% create parallel pool, it usually takes 1 to 6 minutes 

clearvars;
close all;
if isempty(gcp)
    distcomp.feature('LocalUseMpiexec', false);
    c = parcluster;
    pool = parpool(c.NumWorkers);
end
%% create normalized ideal image

imgsz = 64;         % size of normalized ideal image
SRzoom = 8;         % subsampling size of each pixel
Pixelsize = 0.1;    % the pixel size on sample plane, unit is micron
NA = 1.4;           % numerical aperture of the objective lens
Lambda = 0.7;       % emission wavelength of the sample, unit is micron
idealimgnorm = genidealimage(imgsz,Pixelsize,SRzoom,NA,Lambda);

%% select variance map from calibrated map data
% the calibrated maps are 512 by 512 pixels
% varsub:   selected variance map, size is the same as the ideal image size
% gainsub:  selected gain map, size is the same as the ideal image size
% test gain calibration file: gaincalibration_561_gain.mat

gainfile = 'gaincalibration_561_gain.mat';
[varsub,gainsub] = gennoisemap(imgsz,gainfile);
%% generate simulated data
% ims: simulated image from sCMOS camera
% imsd: simulated image from sCMOS camera after gain and offset correction
% imsp: simulated image with Poisson noise only
% imso: simulated ideal image with given photon and background count

N = 5;              % number of simulated data
I = 100;             % total photon count of each emitter on simulated microtubules
bg = 10;             % background photon count
offset = 100;        % offset ADU level of sCMOS camera
[ims,imsd,imsp,imso] = gendatastack(idealimgnorm,varsub,gainsub,I,bg,offset,N);
%% generate noise corrected image
% out: noise corrected image

iterationN = 15;     % number of iterations
alpha = 0.2;         % weight factor of noise contribution only
Rs = 8;              % size of segmented images
[out] = reducenoise(Rs,imsd,varsub,gainsub,imgsz,Pixelsize,NA,Lambda,alpha,iterationN);
h = dipshow(cat(2,imsd,out));
diptruesize(h,400);


