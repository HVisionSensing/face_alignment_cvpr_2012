Author: Roberto Valle Fernández
Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr

### Abstract
The provided project is my own version of Matthias Dantone et al. code based on the CVPR 2012 paper:

Dantone M, Fanelli G, Gall J. and Van Gool L., 
Real Time Facial Feature Detection using Conditional Regression Forest, IEEE Conference on Computer Vision and Pattern Recognition (CVPR'12), 2012.

This package contains the source code for training and evaluation of the 
Conditional Regression Forest. Additional to the source code you can find pretrained trees for head pose estimation and also for facial feature detection. 

### Building
This framework needs the open source computer vision library OpenCV and Boost.

### Demo Application
Running the demo application using the pretrained trees is easy.
```
./demo 0 data/config_ffd.txt data/config_headpose.txt data/haarcascade_frontalface_alt.xml
```

You need to set 4 flags: 
 - mode (0=training, 1=evaluate)
 - path to facial-feature-detection config file
 - path to head-pose config file
 - path to face cascade

<p align="center">
  <img src="http://blog.gimiatlicho.webfactional.com/wp-content/uploads/2012/06/result_web.jpg" alt="Alignment"/>
</p>
