
# Pytorch implementation of SiamFC

- 2020-05-14  my complete code https://github.com/HonglinChu/SiamFC

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
|         SiamFC         |      0.5xx      |

## Reference
```bash

Bertinetto, Luca and Valmadre, Jack and Henriques, Joo F and Vedaldi, Andrea and Torr, Philip H S
		Fully-Convolutional Siamese Networks for Object Tracking
		In ECCV 2016 workshops
		
Refer to the link https://github.com/StrangerZhang/SiamFC-PyTorch
```
