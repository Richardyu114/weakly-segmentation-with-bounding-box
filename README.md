
### Example Dataset:

Sputum Smear--Makerere University, Uganda

- [homepage](http://air.ug/microscopy/)
- [paper](http://proceedings.mlr.press/v56/Quinn16.pdf)
- [code](https://github.com/jqug/microscopy-object-detection/blob/master/CNN%20training%20%26%20evaluation%20-%20tuberculosis.ipynb)

USC Drone

- [homepage](https://github.com/chelicynly/A-Deep-Learning-Approach-to-Drone-Monitoring)
- [paper](https://arxiv.org/abs/1812.08333)

### Requirements:

```
python3
scipy==0.19.0
numpy
pytroch>=1.1.0
torchvision>=0.3.0
PIL
opencv-python
matplotlib
lxml
[pydensecrf](https://github.com/lucasb-eyer/pydensecrf)
```
You can use `pip` to install these packages. Please add `-i https://mirrors.aliyun.com/pypi/simple` after package name if you are in China. 

### Features

**Only support binary pixel classification (one object + background) now!**

#### Model

- FCN
- UNet
- Deeplab v3+

#### Loss

- BCE
- Focal Loss
- Dice Loss
- Lovase Loss

#### Pseudo segmentation label generation

- all bounding box
- inner area of bounding box
- grabcut

#### Dense CRF for post-process

#### Training

- sgd
- adam
- update label for iteration training
- mixup

### Further work

- support multi-classes
- more fast and simple mIoU calculation
- more useful model
- more appropriate optimizer 


### Example of this project:
![GrabCut+FCN+FL](https://raw.githubusercontent.com/Richardyu114/weakly-segmentation-with-bounding-box/master/img/5.png)

![](https://raw.githubusercontent.com/Richardyu114/weakly-segmentation-with-bounding-box/master/img/1.png)

![](https://raw.githubusercontent.com/Richardyu114/weakly-segmentation-with-bounding-box/master/img/2.png)

![](https://raw.githubusercontent.com/Richardyu114/weakly-segmentation-with-bounding-box/master/img/3.png)