# NCS

NCS is a Noise Correction Algorithm for sCMOS cameras. 

# Abstract
Scientific CMOS (sCMOS) cameras are quickly gaining popularity in life sciences, material science and astronomy because its advantages in a much faster frame rate, larger field of view and higher detection efficiency than traditional cameras such as CCD and EMCCD. However, they introduce pixel-dependent noise that generates image artifacts and biases in quantification. 

NCS (noise correction algorithm for sCMOS (CMOS) cameras) is an algorithm that minimizes sCMOS noise from microscopy images with arbitrary structures. In the citation linked below, we show our new method enables significantly reduction of the pixel-dependent noises in fluorescence microscopy using a sCMOS camera and makes its performance approaching that of an ideal camera. 

# NCS source code demo
The demo package consists of functions and scripts written in MATLAB (MathWorks, Natick, MA). The code has been tested in MATLAB version R2015a. To simplify coding, we use [DIPimage toolbox](freely available at http://www.diplib.org/). 

## How to run
To run the demo code for simulated microtubule structure:

	1. Set the current folder in MATLAB to be NCSdemo code\ 
	2. Open script NCSdemo_simulation.m. Set the value of the following parameters: imgsz (image size), Pixelsize, NA (numerical aperture of the objective), Lambda (emission wavelength), I (photon count of each fluorophore), bg (background photon count), N (number of  images), offset (offset ADU level of the sCMOS camera), iterationN (number of iterations), alpha (weight factor of noise contribution), Rs (size of segmented image) and the type of the OTF mask.
	3. Run the code. 
	4. The output includes: imsd (sCMOS image stack), out (the noise corrected image).
	5. The computation time depends on the imgsz (image size), N (number of images), iterationN (number of iterations) and Rs (segmentation size). 
  
The usage of the demo code for experimental data is similar to the instruction above. Please see the included script, NCSdemo_experiment.m, for detail. 

## License and Citation
NCS is released under the [GNU license](https://github.com/HuanglabPurdue/NCS/edit/master/LICENSE).

Please cite NCS in your publications if it helps your research:

  	 @article{Liu2017NCS,
	Author = {Liu, Sheng and Mlodzianoski1, Michael J. and Hu, Zhenhua and Ren, Yuan and McElmurry, Kristi and Suter, Daniel M. and Huang, Fang},
	Journal = {Nature Methods},
	Title = {sCMOS noise-correction algorithm for microscopy images},
	Year = {2017}
	volume = {14}
	number = {8}
	pages = {760-761}
   }
