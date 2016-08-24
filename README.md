# Fast Optical Flow using Dense Inverse Search (DIS) #

Our code is released only for scientific or personal use.
Please contact us for commercial use.

If used this work, please cite:

`@inproceedings{kroegerECCV2016,
   Author    = {Till Kroeger and Radu Timofte and Dengxin Dai and Luc Van Gool},
   Title     = {Fast Optical Flow using Dense Inverse Search},
   Booktitle = {Proceedings of the European Conference on Computer Vision ({ECCV})},
   Year      = {2016}} `

Is you use the variational refinement, please additionally cite:

` @inproceedings{weinzaepfelICCV2013,
    TITLE = {{DeepFlow: Large displacement optical flow with deep matching}},
    AUTHOR = {Weinzaepfel, Philippe and Revaud, J{\'e}r{\^o}me and Harchaoui, Zaid and Schmid, Cordelia},
    BOOKTITLE = {{ICCV 2013 - IEEE International Conference on Computer Vision}},
    YEAR = {2013}} `

  
  
  
## Compiling ##

The program was only tested under a 64-bit Linux distribution.
SSE instructions from built-in X86 functions for GNU GCC were used.

The following will build four binaries: 
Two for optical flow (`run_OF_*`) and two for depth from stereo (`run_DE_*`).
For each problem, a fast variant operating on intensity images (`run_*_INT`) and 
a slower variant operating on RGB images (`run_*_RGB`) is provided.

```
mkdir build
cd build
cmake ../
make -j
```

The code depends on Eigen3 and OpenCV. However, OpenCV is only used for image loading, 
scaling and gradient computation (`run_dense.cpp`). It can easily be replaced by other libraries.
      
      
      
      
## Usage ##
The interface for all four binaries (`run_*_*`) is the same.

VARIANT 1 (Uses operating point 2 of the paper, automatically selects coarsest scale):

` ./run_*_* image1.png image2.png outputfile `


VARIANT 2 (Manually select operating point X=1-4, automatically selects coarsest scale):

`  ./run_*_* image1.png image2.png outputfile X `


VARIANT 3 (Set all parameters explicitly):

` ./run_*_* image1.png image2.png outputfile p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20`

Example for variant 3 using operating point 2 of the paper:

` ./run_OF_INT in1.png int2.png out.flo 5 3 12 12 0.05 0.95 0 8 0.40 0 1 0 1 10 10 5 1 3 1.6 2  `



Parameters:
```
1. Coarsest scale                               (here: 5)
2. Finest scale                                 (here: 3)
3/4. Min./Max. iterations                       (here: 12)
5./6./7. Early stopping parameters
8. Patch size                                   (here: 8)
9. Patch overlap                                (here: 0.4)
10.Use forward-backward consistency             (here: 0/no)
11.Mean-normalize patches                       (here: 1/yes)
12.Cost function                                (here: 0/L2)  Alternatives: 1/L1, 2/Huber, 10/NCC
13.Use TV refinement                            (here: 1/yes)
14./15./16. TV parameters alpha,gamma,delta     (here 10,10,5)
17. Number of TV outer iterations               (here: 1)
18. Number of TV solver iterations              (here: 3)
19. TV SOR value                                (here: 1.6)
20. Verbosity                                   (here: 2) Alternatives: 0/no output, 1/only flow runtime, 2/total runtime
```


The optical flow output is saves as .flo file.
(http://sintel.is.tue.mpg.de/downloads)

The interface for depth from stereo is exactly the same. The output is saves as pfm file.
(http://vision.middlebury.edu/stereo/code/)


NOTES:
1. For better quality, increase the number iterations (param 3/4), use finer scales (param. 2), higher patch overlap (param. 9), more outer TV iterations (param. 17)
2. L1/Huber cost functions (param. 12) provide better results, but require more iterations (param. 3/4)



## Bugs and extensions ##

If you find bugs, etc., please feel free to contact me.
Contact details are available on my webpage.
http://www.vision.ee.ethz.ch/~kroegert/



## History ##

July 2016 	v1.0.0 - Initial Release
August 2016 	v1.0.1 - Minor Bugfix: Error in L1 and Huber error norm computation.




## LICENCE CONDITIONS ##

GPLv3: http://gplv3.fsf.org/

All programs in this collection are free software: 
you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.











