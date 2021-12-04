# Ultrafast water-fat separation using deep learning-based single-shot MRI

Author: Xinran Chen, Lin Chen\*

Email:  xinranchen@stu.xmu.edu.cn   chenlin21@xmu.edu.cn

Affiliations:
Department of Electronic Science, Fujian Provincial Key Laboratory of Plasma and Magnetic Resonance, School of Electronic Science and Engineering, National Model Microelectronics College, Xiamen University, Xiamen, Fujian, China



This toolbox contains code that implement the water-fat separation described in paper:

Xinran Chen, Wei Wang, Jianpan Huang, Jian Wu, Lin Chen\*, Congbo Cai, Shuhui Cai, Zhong Chen\*, **Ultrafast water-fat separation using deep learning-based single-shot MRI**  （under revision）



**Due to the size limitation of Github, the  files "\models\UNet_trained_for_fat\model.ckpt-150000.data-00000-of-00001"  and "\models\UNet_trained_for_water\model.ckpt-150000.data-00000-of-00001"  were removed.** 

**The full version can be downloaded from zenodo platform (DOI:10.5281/zenodo.5758520)** :

<a href="https://doi.org/10.5281/zenodo.5758520"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.5758520.svg" alt="DOI"></a>



Software requirements

MATLAB:MATLAB vision: 9.9.0 (R2020b)

Python:
Python 3.6

Numpy 1.19.1

TensorFlow 1.8.0

Scipy 1.4.1

Scikit-image 0.17.2

Tflearn 0.3.2

**Instruction**:

**Pre-processing for experimental data**:The data for testing was saved as .Charles format. Its dimension is 128\*128\*10. 128\*128 is the size of input and label image, while 10 channels contain two channels for the real (channel 1) and imaginary part (channel 2) of input images, two channels for the label images (in experimental data they are also blank) of water (channel 3) and fat (channel 4). 

**Put data into trained U-net for water-fat separation**: The neural networks have been trained for reconstruct water and fat, respectively.  The trained model for water/fat is provided in “WFSR_SPEN\models”. To use the weights, you can run:

My_Interface_WFSR.py

Or call the python command in the MATLAB interface as annotated in the main function. Notice that Python3.6 is only Compatible with MATLAB versions from R2017b to R2020b.
The inference results are stored in “reconstruction_result\DL_result” folder as a .mat file.

**References**:

**CG:** Schmidt R, Frydman L. In vivo 3D spatial/1D spectral imaging by spatiotemporal encoding: a new single-shot experimental and processing approach. Magn Reson Med. 2013;70(2):382-391.

**SWAF:** Huang J, Chen L, Chan KWY, Cai C, Cai S, Chen Z. Super-resolved water/fat image reconstruction based on single-shot spatiotemporally encoded MRI. J Magn Reson. 2020;314:106736.


Welcome your comments and suggestions.

For more information, please visit: https://linchenmri.com



Dec. 4, 2021
