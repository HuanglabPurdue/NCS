%------ Demo code for noise correction algorithm for sCMOS camera (NCS algorithm) on experimental data------------
% software requirement: Matlab R2015a or later
%                       Dipimage toolbox 2.7 or later
% system requirement:   CPU Intel Core i7
%                       32 GB RAM 
% Data format: data must be .mat file, containing a R by R by N matrix, R
%              is the x and y dimension of the image, N is the number of
%              images
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, November 2017

%% setup parallel pool, it usually takes 1 to 6 minutes 

clearvars;close all
if isempty(gcp)
    distcomp.feature( 'LocalUseMpiexec', false );
    c = parcluster;
    pool = parpool(c.NumWorkers);
end
%% select data and gain calibration file
% test gain calibration file: gaincalibration_561_gain.mat
% test data file:             EB300004.mat

gainfile = 'gaincalibration_561_gain.mat';
datafile = 'EB300004.mat';
load(gainfile)
datald = load(datafile);
name = fields(datald);
data = datald.(name{1});
%% crop region of interest (ROI)
sz = 256;                       % ROI size
rect = [59,58;sz,sz];           % The [left, top; width, height] of the selected ROI
subims = cropimage(data,sz,rect);
subvar = cropimage(ccdvar,sz,rect);
suboffset = cropimage(ccdoffset,sz,rect);
subgain = cropimage(gain,sz,rect);
subgain(subgain<1) = 1;         % ensure the pixel value from the selected gain map is greater than 1.

%% apply gain and offset correction
N = size(subims,3);             % number of sCMOS images
suboffsetL = repmat(suboffset,[1,1,N]);
subgainL = repmat(subgain,[1,1,N]);
subvarL = repmat(subvar,[1,1,N]);
imsd = (subims-suboffsetL)./subgainL;
imsd0 = imsd;                   % gain and offset corrected images
imsd(imsd<=0) = 1e-6;           % gain and offset corrected images, remove pixels with negative value
%% generate noise corrected image
% The test data contains 20 frames.
% The user could select a subset of the data by running the following lines instead

Rs = 1;                         % size of segmented images                        
alpha = 1;                      % weight factor of noise contribution only
Pixelsize = 0.091;              % the pixel size on sample plane, unit is micron
Lambda = 0.54;                  % emission wavelength of the sample, unit is micron
NA = 1.35;                      % numerical aperture of the objective lens
iterationN = 15;                % number of iterations
Nsub = 1;                       % number of images from the subset of the data
[out] = reducenoise(Rs,imsd(:,:,1:Nsub),subvar,subgain,sz,Pixelsize,NA,Lambda,alpha,iterationN);
imgcom = cat(2,imsd(:,:,1:Nsub),out);
h = dipshow(imgcom);
diptruesize(h,200);




