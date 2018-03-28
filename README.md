# Kaggle-2018-data-bowl
Record of the competition
## 1.Data Pre-Processing
#### Histogram Equalization
It makes the image constrast more obvious. It maybe helpful in preprocessing.</br>
![image](https://github.com/mjDelta/Kaggle-2018-data-bowl/blob/master/imgs/hist.png)</br>
#### Cropping Strategy
This strategy is aimmed at seamless segmentation of arbitrary large images. Missing input data is extrapolated by mirroring.</br>
![image](https://github.com/mjDelta/Kaggle-2018-data-bowl/blob/master/imgs/cropping.png)</br>
## 2.Deep Learning Models
#### U-net
U-Net is the basic Network in this competition. It has played an import part in many projects. Here, the number of filters is reduced for fasting training.</br>
#### M-net
M-net is proposed in Jan,2018 by Huazhu Fu. It adds some structure on the base of U-net and tries to solve the information loss in pooling and upsampling.</br>
![image](https://github.com/mjDelta/Kaggle-2018-data-bowl/blob/master/imgs/mnet.PNG)</br>
## 3.Results Display
Here are some of the segmentation results. The red line draws the segmentation results of the colorful images after gray scaling.</br>
![image](https://github.com/mjDelta/Kaggle-2018-data-bowl/blob/master/imgs/result.PNG)</br>
![image](https://github.com/mjDelta/Kaggle-2018-data-bowl/blob/master/imgs/result2.PNG)</br>
