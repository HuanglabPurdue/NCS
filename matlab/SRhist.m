% SRHIST   Generate a 2D histogram image from a set of x,y coordinates
%
% SYNOPSIS:
%   histim=SRhist(xsz,ysz,xco,yco)
%
% INPUTS:
%   xsz
%       x dimension of the histogram image, in pixels
%   ysz
%       y dimension of the histogram image, in pixels
%   xco
%       x coordinates, in pixels
%   yco
%       y coordinates, in pixels
%
% NOTES:
%   The pixel value of the histogram image is equal to the number of
%   coordinates fall within that pixel
%
% (C) Copyright 2017                The Huang Lab
%     All rights reserved           Weldon School of Biomedical Engineering
%                                   Purdue University
%                                   West Lafayette, Indiana
%                                   USA
% Fang Huang, April 2016

function histim=SRhist(xsz,ysz,xco,yco)
xco=xco(:)+1;
yco=yco(:)+1;

histim=zeros(xsz,ysz);

frm=10000;

seg=ceil(numel(xco)/frm);

for ss=1:1:seg
    st=(ss-1)*frm+1;
    if ss==seg
        ed=numel(xco);
    else
        ed=ss.*frm;
    end
    
    tmpx=floor(xco(st:ed));
    tmpy=floor(yco(st:ed));
    
    mask=tmpx<=xsz&tmpy<=ysz&tmpx>0&tmpy>0;
    currx=tmpx(mask);
    curry=tmpy(mask);
    
    idx = sub2ind(size(histim), currx, curry);
    [y]=countrepeats(idx);
    histim(idx)=histim(idx)+y;
end

function [out]=countrepeats(in)
out = arrayfun(@(z)nnz(in==z), in);