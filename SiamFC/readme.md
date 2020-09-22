
# Pytorch implementation of SiamFC
## Description
[My Code-VID](https://github.com/HonglinChu/SiamFC_VID)
[My Code-GOT](https://github.com/HonglinChu/SiamFC_GOT)

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

## Benchmark results

#### OTB100

|         Tracker        |       AUC       |
| ---------------------- | --------------- |
|         SiamFC         |      0.57x     |

## Reference
```bash

[1] Bertinetto L, Valmadre J, Henriques J F, et al. Fully-convolutional siamese networks for object tracking. European conference on computer vision. Springer, Cham, 2016: 850-865.
		
[2] https://github.com/StrangerZhang/SiamFC-PyTorch
```
