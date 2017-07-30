%----- Example code for using multiple gain and variance maps -----
% reference: Liu,Sheng,et al.,sCMOS noise-correction algorithm for microscopy images,Nature Methods 14,760-761(2017)
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Sheng Liu, June 2017
%%
R = 128;                    % simulated image size
lightlevel = [1500,3000];   % devision points of the ADU levels, here the number of the ADU levels is 3
% simulate a stack of gain maps, gain, it is a R x R x 3 matrix
gain = cat(3,ones(R,R).*1.5, ones(R,R).*2,ones(R,R).*3);
% simulate a stack of variance maps, var, it is a R x R x 3 matrix
var = cat(3,ones(R,R).*10,ones(R,R).*100,ones(R,R).*1000);
% simulate a data stack, ims, it is a R x R x N matrix, N is a integer
[X,Y] = meshgrid([-R/2+1:R/2],[-R/2+1:R/2]);
Z = sqrt(X.^2+Y.^2)./R.*sqrt(2);
ims = cat(3,Z.*3500,Z.*4000,Z.*4500,Z.*5000);
% generate a gain and a variance stack with the same size of the data stack
[gainstack,varstack] = genmaps(ims,gain,var,lightlevel);
% show the generated gain and variance stack
dipshow(gainstack)
dipshow(varstack)