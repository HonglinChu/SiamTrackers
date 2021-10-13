# SiamFC

## Description

- [x] Support VSCode debug

- [x] Support train and test

- [x] Support 9 datasets evaluation

## Train
```bash
cd  /home/xxx/xxx/SiamTrackers-master/2-SiamFC/SiamFC-GOT 
python ./bin/train.py
```
## Test
```bash
cd  /home/xxx/xxx/SiamTrackers-master/2-SiamFC/SiamFC-GOT 
python ./bin/test.py
```

## Dataset

you should download GOT-10k dataset [BaiduYun](https://pan.baidu.com/s/172oiQPA_Ky2iujcW5Irlow) password: uxds

## Model

[BaiduYun](https://pan.baidu.com/s/1dnsB5MzVTKMzBFr7_-6C4Q) password: wxwh

##  Experiment
| Dataset       |  OTB2015         |     DTB70        | UAV20L    | UAV123 |UAVDT|
|:-----------   |:----------------:|:----------------:|:--------:|:------:|:-----:|
| Success Score       | 0.570            |  0.487        |0.410|0.504|0.451|
| Precision Score     | 0.767           |  0.735         |0.566|0.702|0.710|

## Reference

[Huang Lianghua](https://github.com/huanglianghua/siamfc-pytorch)
```

[1] Bertinetto L, Valmadre J, Henriques J F, et al. Fully-convolutional siamese networks for object tracking. European conference on computer vision. Springer, Cham, 2016: 850-865.
		
```



