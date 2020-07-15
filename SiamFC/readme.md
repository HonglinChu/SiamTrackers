
# Pytorch implementation of SiamFC
## Description
My code https://github.com/HonglinChu/SiamFC

## File Tree
```
├── bin
├── got10k
└── siamfc
```
## Data preparation

```bash
you should download ILSVRC2015_VID dataset first
- python bin/create_dataset.py
- python bin/create_lmdb.py
```

## Training
```bash
python train_siamfc.py
```
## Testing
```bash
python test_siamfc.py
```

## Benchmark results

#### OTB100

|         Tracker        |       AUC       |
| ---------------------- | --------------- |
|         SiamFC         |      0.57x     |

## Reference
```bash

Bertinetto, Luca and Valmadre, Jack and Henriques, Joo F and Vedaldi, Andrea and Torr, Philip H S
		Fully-Convolutional Siamese Networks for Object Tracking
		In ECCV 2016 workshops
		
Refer to the link https://github.com/StrangerZhang/SiamFC-PyTorch
```
