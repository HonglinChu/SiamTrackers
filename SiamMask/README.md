# Pytorch implementation of SiamMask

## Description
[Official Code](https://github.com/foolwood/SiamMask)

![image](../image/siammask1.png)
## Dataset

- **ILSVRC2015 VID** [BaiDuYun](https://pan.baidu.com/s/1CXWgpAG4CYpk-WnaUY5mAQ) password: uqzj 

- **ILSVRC2015 DET** [BaiDuYun](https://pan.baidu.com/s/1t2IgiYGRu-sdfOYwfeemaQ) password: 6fu7 

- **YTB-Crop511** [BaiduYun](https://pan.baidu.com/s/112zLS_02-Z2ouKGbnPlTjw) password: ebq1 

- **COCO** [BaiduYun](https://pan.baidu.com/s/17AMGS2ezLVd8wFI2NbJQ3w) password: ggya 

- **YTB-VOS** [BaiduYun](https://pan.baidu.com/s/1WMB0q9GJson75QBFVfeH5A) password: sf1m 

- **DAVIS2017** [BaiduYun](https://pan.baidu.com/s/1JTsumpnkWotEJQE7KQmh6A) password: c9qp

## Experiment
My experiment results on VOT2016 and VOT2018
|      ||     VOT16-E   |         VOT16-A   |        VOT16-R |    |    VOT18-E    |       VOT18-A |        VOT18-R|
|:---------:|:-----:|:--------:| :------:    |:------:  |:------:   |:------: |:------:|:------:|
| SiamMask-box  |   | 0.414    |  0.622      |   0.224   |           |0.363  |   0.585  | 0.300  |
|  SiamMask     |   |  0.433   |   0.639     |   0.205   |          | 0.380   |  0.610   |0.281 |   
| SiamMask-LD  |    | 0.456    |   0.622     |    0.244  |          | 0.421     | 0.598  | 0.234|

## Reference
```
[1] Wang Q, Zhang L, Bertinetto L, et al. Fast online object tracking and segmentation: A unifying approach. Proceedings of the IEEE conference on computer vision and pattern recognition. 2019: 1328-1338.

[2] https://github.com/foolwood/SiamMask

```
