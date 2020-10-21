# SiamFC

# SiamFC

## Description

- [x] Support VSCode debug

- [x] Support train and test

- [x] Support 9 datasets evaluation

## File Tree
```
├── bin
├── got10k
└── siamfc
```

## Dataset

you should download GOT-10k dataset [BaiduYun](https://pan.baidu.com/s/172oiQPA_Ky2iujcW5Irlow) password: uxds

## Train
```bash
python ./bin/train.py
```
## Test
```bash
python ./bin/test.py
```

## Model

[BaiduYun](https://pan.baidu.com/s/1dnsB5MzVTKMzBFr7_-6C4Q) password: wxwh

##  Experiment
| Dataset       |  OTB2015         |     DTB70        | UAV20L    | UAV123 |UAVDT|
|:-----------   |:----------------:|:----------------:|:--------:|:------:|:-----:|
| Success Score       | 0.570            |  0.487        |0.410|0.504|0.451|
| Precision Score     | 0.767           |  0.735         |0.566|0.702|0.710|

## Reference
```bash

[1] Bertinetto L, Valmadre J, Henriques J F, et al. Fully-convolutional siamese networks for object tracking. European conference on computer vision. Springer, Cham, 2016: 850-865.
		
[2] https://github.com/StrangerZhang/SiamFC-PyTorch (The results of siamfc with VID dataset)    

[3] https://github.com/huanglianghua/siamfc-pytorch (Recommend!!! The  results of siamfc with GOT dataset are better than official )
```



