
# Pytorch implementation of SiamFC
## Description

[My Code](https://github.com/HonglinChu/SiamFC)


## File Tree
```
├── bin
├── got10k
└── siamfc
```
## Data preparation
you should download [ILSVRC2015_VID](https://pan.baidu.com/s/1Bu7yOxjM_ByOF_RoSWmmOw) password: fj43 

```

python bin/create_dataset.py

python bin/create_lmdb.py
```

## Training
```bash
python ./bin/train_siamfc.py
```
## Testing
```bash
python ./bin/test_siamfc.py
```
## Model

[BaiduYun](https://pan.baidu.com/s/1r2CASAg5bMWsTAS_ODQVZA) password: sej8

## Benchmark results

#### OTB100

|         Tracker        |       AUC       |
| ---------------------- | --------------- |
|         SiamFC         |      0.57x     |

## Reference
```bash

[1] Bertinetto L, Valmadre J, Henriques J F, et al. Fully-convolutional siamese networks for object tracking. European conference on computer vision. Springer, Cham, 2016: 850-865.
		
[2] https://github.com/StrangerZhang/SiamFC-PyTorch

[3] https://github.com/huanglianghua/siamfc-pytorch (Recommend!!! the siamfc code with GOT dataset is better than official )
```
