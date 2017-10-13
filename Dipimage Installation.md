# Installation of dipimage on Windows System
  1.	Download dipimage from [DIPimage](http://www.diplib.org/download). Choose automatic installation package. 
  2.	After installation, create a Matlab script called startup.m file under the folder C:\Users\XXX\Documents\MATLAB\, copy the following lines in the startup.m file and save the file:
  
      addpath('C:\Program Files\DIPimage 2.8(your dipimage version)\common\dipimage');
      dip_initialise;
      dipfig -unlink
      dipsetpref('DefaultMappingMode','lin');
      dipsetpref('DebugMode','on');
      dipsetpref('TrueSize','off');

  3.	Restart Matlab, Matlab will execute the startup.m automatically during each startup. If the dipimage toolbox is installed correctly, the following lines should show up in the command window:

      DIPlib 2.8 (Nov 11 2015 - Development [on Cygwin (with OpenMP)])
          Scientific Image Analysis Library
          Quantitative Imaging Group, Delft University of Technology 1995-2015
          info@diplib.org


      dipIO 2.8 (Nov 11 2015 - Development)
          File I/O library for DIPlib
          Quantitative Imaging Group, Delft University of Technology 1999-2015
          info@diplib.org
