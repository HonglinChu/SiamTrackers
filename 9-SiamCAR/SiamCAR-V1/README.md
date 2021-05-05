# SiamCAR-V1

[Official Code](https://github.com/ohhhyeahhh/SiamCAR)

[哔哩哔哩](https://www.bilibili.com/video/BV1N5411Y7cA?from=search&seid=1639036191308901519)

## Train

```
We only use GOT-10k to train siamcar network

sh ./bin/cmd_dist_train.sh 

```

## Test
```
sh ./bin/cmd_test.sh 
```

## Results
```
Speed: 255.8fps
checkpoint_e25.pth VOT2018 lr-0.35  pk-_0.2  win-_0.45
------------------------------------------------------------
|Tracker Name| Accuracy | Robustness | Lost Number |  EAO  |
------------------------------------------------------------
|  siamcar   |  0.576   |   0.356    |    76.0     | 0.283 |
------------------------------------------------------------

checkpoint_e27.pth  VOT2018 lr-0.35  pk-_0.2  win-_0.45
------------------------------------------------------------
|Tracker Name| Accuracy | Robustness | Lost Number |  EAO  |
------------------------------------------------------------
|  siamcar   |  0.575   |   0.361    |    77.0     | 0.292 |
------------------------------------------------------------

checkpoint_e46.pth VOT2018  lr-0.35  pk-_0.2  win-_0.45
------------------------------------------------------------
|Tracker Name| Accuracy | Robustness | Lost Number |  EAO  |
------------------------------------------------------------
|  siamcar   |  0.576   |   0.356    |    76.0     | 0.290 |
------------------------------------------------------------

checkpoint_e39.pth VOT2018  lr-0.35  pk-_0.2    win -_0.45
------------------------------------------------------------
|Tracker Name| Accuracy | Robustness | Lost Number |  EAO  |
------------------------------------------------------------
|  siamcar   |  0.567   |   0.384    |    82.0     | 0.286 |
------------------------------------------------------------

hp_search load checkpoint_e27.pth  VOT2018 lr-0.35  pk-_0.2  win-_0.45 
------------------------------------------------------------------------------------------------
|                  Tracker Name                  | Accuracy | Robustness | Lost Number |  EAO  |
------------------------------------------------------------------------------------------------
| checkpoint_e27_r255_pk-0.120_wi-0.450_lr-0.320 |  0.571   |   0.309    |    66.0     | 0.318 |
| checkpoint_e27_r255_pk-0.250_wi-0.440_lr-0.320 |  0.567   |   0.300    |    64.0     | 0.317 |
| checkpoint_e27_r255_pk-0.140_wi-0.440_lr-0.300 |  0.569   |   0.300    |    64.0     | 0.317 |
| checkpoint_e27_r255_pk-0.150_wi-0.450_lr-0.320 |  0.567   |   0.314    |    67.0     | 0.316 |
| checkpoint_e27_r255_pk-0.160_wi-0.450_lr-0.320 |  0.569   |   0.314    |    67.0     | 0.315 |
| checkpoint_e27_r255_pk-0.150_wi-0.440_lr-0.320 |  0.568   |   0.309    |    66.0     | 0.314 |
| checkpoint_e27_r255_pk-0.160_wi-0.460_lr-0.320 |  0.569   |   0.347    |    74.0     | 0.313 |
| checkpoint_e27_r255_pk-0.150_wi-0.480_lr-0.340 |  0.570   |   0.337    |    72.0     | 0.311 |
| checkpoint_e27_r255_pk-0.150_wi-0.440_lr-0.300 |  0.568   |   0.318    |    68.0     | 0.310 |
| checkpoint_e27_r255_pk-0.280_wi-0.450_lr-0.350 |  0.571   |   0.328    |    70.0     | 0.309 |
| checkpoint_e27_r255_pk-0.240_wi-0.440_lr-0.400 |  0.570   |   0.318    |    68.0     | 0.309 |
| checkpoint_e27_r255_pk-0.120_wi-0.450_lr-0.360 |  0.571   |   0.323    |    69.0     | 0.309 |
| checkpoint_e27_r255_pk-0.150_wi-0.460_lr-0.340 |  0.567   |   0.328    |    70.0     | 0.309 |
| checkpoint_e27_r255_pk-0.240_wi-0.440_lr-0.380 |  0.568   |   0.328    |    70.0     | 0.308 |
| checkpoint_e27_r255_pk-0.180_wi-0.440_lr-0.350 |  0.573   |   0.318    |    68.0     | 0.308 |
------------------------------------------------------------------------------------------------
```
# Reference
```
https://github.com/ohhhyeahhh/SiamCAR

[1] Guo D ,  Wang J ,  Cui Y , et al. SiamCAR: Siamese Fully Convolutional Classification and Regression for Visual Tracking[C]//CVPR,2020.

```
