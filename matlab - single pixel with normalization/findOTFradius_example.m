%----- Example code for determining the OTF radius from the experimental data---
% Data format: data must be .mat file, containing a R by R by N matrix, R
%              is the x and y dimension of the image, N is the number of
%              images
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, June 2017

%% load the data
load('beaddata.mat')
sz = size(ims,1);
Pixelsize = 0.091;
[freqmax,otf,ind] = findOTFradius(ims,Pixelsize);
%% plot the overlay of the image in Fourier space and the found OTF boundary 
h = figure;
h.Position = [800,660,400,400];
ha = axes;
ha.Position = [0,0,1,1];
ha.YDir = 'reverse';
hold(ha,'on');
imagesc(otf,[-1,2])
colormap(grey)
rf = ind;
theta = linspace(0,2*pi,200);
cc = sz/2+0.5;
plot(rf.*cos(theta)+cc,rf.*sin(theta)+cc,'r-','linewidth',2,'parent',ha)
axis equal
axis off
